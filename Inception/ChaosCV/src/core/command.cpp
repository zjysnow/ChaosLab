#include "core/command.hpp"
#include "core/pipeline.hpp"

#include "dnn/option.hpp"

#include <vulkan/vulkan.hpp>

namespace chaos
{
	Command::Command(const VulkanDevice* vkdev) : vkdev(vkdev) {}
	Command::~Command() {}

	ComputeCommand::ComputeCommand(const VulkanDevice* vkdev) : Command(vkdev)
	{
		VkResult ret;
		VkCommandPoolCreateInfo pool_info{};
		pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		pool_info.queueFamilyIndex = vkdev->info.compute_queue_family_index;
		pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

		ret = vkCreateCommandPool(vkdev->GetDevice(), &pool_info, nullptr, &command_pool);
		CHECK_EQ(VK_SUCCESS, ret) << "vkCreateCommandPool failed " << ret;

		VkCommandBufferAllocateInfo command_buffer_allocate_info{};
		command_buffer_allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		command_buffer_allocate_info.commandPool = command_pool;
		command_buffer_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		command_buffer_allocate_info.commandBufferCount = 1;
		ret = vkAllocateCommandBuffers(vkdev->GetDevice(), &command_buffer_allocate_info, &command_buffer);
		CHECK_EQ(VK_SUCCESS, ret);

		VkFenceCreateInfo fence_create_info{};
		fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		ret = vkCreateFence(vkdev->GetDevice(), &fence_create_info, 0, &fence);
		CHECK_EQ(VK_SUCCESS, ret);

		VkCommandBufferBeginInfo command_buffer_begin_info{};
		command_buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		command_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		ret = vkBeginCommandBuffer(command_buffer, &command_buffer_begin_info);
		CHECK_EQ(VK_SUCCESS, ret);

		if (vkdev->info.support_VK_KHR_descriptor_update_template)
		{
			vkCmdPushDescriptorSetWithTemplate = (PFN_vkCmdPushDescriptorSetWithTemplateKHR)vkGetDeviceProcAddr(vkdev->GetDevice(), "vkCmdPushDescriptorSetWithTemplateKHR");
		}
	}
	ComputeCommand::~ComputeCommand()
	{
		Release();
	}
	void ComputeCommand::Release()
	{
		if (fence)
		{
			vkDestroyFence(vkdev->GetDevice(), fence, nullptr);
			fence = nullptr;
		}
		if (command_pool)
		{
			vkDestroyCommandPool(vkdev->GetDevice(), command_pool, nullptr);
			command_pool = nullptr;
		}
	}
	void ComputeCommand::RecordUpload(const Tensor& src, VulkanTensor& dst, const Option& opt)
	{
		if (dst.empty()) dst.CreateLike(src, opt.blob_vkallocator);
		if (dst.allocator->mappable)
		{
			memcpy(dst.mapped_data(), src.data, src.total() * src.depth * src.packing);
		}
		else
		{
			VulkanTensor staging;
			staging.CreateLike(src, opt.staging_vkallocator);
			memcpy(staging.mapped_data(), src.data, src.total() * src.depth * src.packing);

			staging.data->access_flag = VK_ACCESS_HOST_WRITE_BIT;
			staging.data->stage_flag = VK_PIPELINE_STAGE_HOST_BIT;

			RecordClone(staging, dst, opt);

			buffers.push_back(staging);
		}
	}
	void ComputeCommand::RecordDownload(const VulkanTensor& src, Tensor& dst, const Option& opt)
	{
		if (dst.empty()) dst.CreateLike(src, opt.blob_allocator);
		if (src.allocator->mappable)
		{
			memcpy(dst.data, src.mapped_data(), src.total() * src.depth * src.packing);
		}
		else
		{
			VulkanTensor staging;
			staging.CreateLike(src, opt.staging_vkallocator);

			RecordClone(src, staging, opt);

			buffers.push_back(staging);
			download_post.push_back(dst);

			Record r;
			r.type = Record::TYPE_DOWNLOAD;
			r.post_download.src = (uint32)buffers.size() - 1;
			r.post_download.dst = (uint32)download_post.size() - 1;
			delayed_records.push_back(r);
		}
	}

	void ComputeCommand::RecordClone(const VulkanTensor& src, VulkanTensor& dst, const Option& opt)
	{
		if (dst.empty()) dst.CreateLike(src, opt.blob_vkallocator);
		if (src.data->access_flag & VK_ACCESS_TRANSFER_WRITE_BIT || src.data->stage_flag != VK_PIPELINE_STAGE_TRANSFER_BIT)
		{
			// barrier device any @ compute to transfer-read @ compute
			VkBufferMemoryBarrier barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
			barrier.srcAccessMask = src.data->access_flag;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.buffer = src.buffer();
			barrier.offset = src.buffer_offset();
			barrier.size = src.buffer_capacity();

			VkPipelineStageFlags src_stage = src.data->stage_flag;
			VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

			vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);

			// mark device transfer-read @ transfer
			src.data->access_flag = VK_ACCESS_TRANSFER_READ_BIT;
			src.data->stage_flag = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}

		// mark device transfer - write @ transfer
		dst.data->access_flag = VK_ACCESS_TRANSFER_WRITE_BIT;
		dst.data->stage_flag = VK_PIPELINE_STAGE_TRANSFER_BIT;

		VkBufferCopy region{};
		region.srcOffset = src.buffer_offset();
		region.dstOffset = dst.buffer_offset();
		region.size = std::min(src.buffer_capacity(), dst.buffer_capacity());

		vkCmdCopyBuffer(command_buffer, src.buffer(), dst.buffer(), 1, &region);
	}
	

	void ComputeCommand::RecordPipeline(const ComputePipeline* pipeline, const std::vector<VulkanTensor>& buffer_bindings, const std::vector<VulkanConstantType>& constants, const Shape& dispatcher)
	{
		for (const auto& binding : buffer_bindings)
		{
			if (binding.data->access_flag & VK_ACCESS_SHADER_WRITE_BIT || binding.data->stage_flag != VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
			{
				// barrier device any @ compute/null to shader-readwrite @ compute
				VkBufferMemoryBarrier barrier{}; // = new VkBufferMemoryBarrier[1];
				barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
				barrier.srcAccessMask = binding.data->access_flag;
				barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
				barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				barrier.buffer = binding.buffer();
				barrier.offset = binding.buffer_offset();
				barrier.size = binding.buffer_capacity();

				VkPipelineStageFlags src_stage = binding.data->stage_flag;
				VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

				vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);

				binding.data->access_flag = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
				binding.data->stage_flag = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

				buffers.push_back(binding);
			}
		}

		// record bind pipeline
		vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);

		std::vector<VkDescriptorBufferInfo> descriptor_buffer_infos(buffer_bindings.size());
		for (size_t i = 0; const auto& binding : buffer_bindings)
		{
			descriptor_buffer_infos[i].buffer = binding.buffer();
			descriptor_buffer_infos[i].offset = binding.buffer_offset();
			descriptor_buffer_infos[i].range = binding.total() * binding.depth * binding.packing;
			i++;
		}
		
		if (vkdev->info.support_VK_KHR_descriptor_update_template)
		{
			vkCmdPushDescriptorSetWithTemplate(command_buffer, pipeline->descriptor_update_template, pipeline->pipeline_layout, 0, descriptor_buffer_infos.data());
		}
		else
		{
			// should use descritor pool and sets
		}

		// push constant
		vkCmdPushConstants(command_buffer, pipeline->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32)(constants.size() * sizeof(VulkanConstantType)), constants.data());
		//
		uint32 group_count_x = (dispatcher[2] + pipeline->local_size_x - 1) / pipeline->local_size_x;
		uint32 group_count_y = (dispatcher[1] + pipeline->local_size_y - 1) / pipeline->local_size_y;
		uint32 group_count_z = (dispatcher[0] + pipeline->local_size_z - 1) / pipeline->local_size_z;
		vkCmdDispatch(command_buffer, group_count_x, group_count_y, group_count_z);
	}

	void ComputeCommand::SubmitAndWait()
	{
		VkResult ret;
		ret = vkEndCommandBuffer(command_buffer);
		CHECK_EQ(VK_SUCCESS, ret);

		VkQueue compute_queue = vkdev->AcquireQueue(vkdev->info.compute_queue_family_index);
		CHECK_NE(nullptr, compute_queue) << "out of compute queue";

		VkSubmitInfo submit_info{};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; 
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffer;

		ret = vkQueueSubmit(compute_queue, 1, &submit_info, fence);
		CHECK_EQ(VK_SUCCESS, ret) << "vkQueueSumbit failed " << ret;

		vkdev->ReclaimQueue(vkdev->info.compute_queue_family_index, compute_queue);

		ret = vkWaitForFences(vkdev->GetDevice(), 1, &fence, VK_TRUE, (uint64)-1);
		CHECK_EQ(VK_SUCCESS, ret) << "vkWaitForFences failed " << ret;

		for (const auto& record : delayed_records)
		{
			switch (record.type)
			{
			case Record::TYPE_DOWNLOAD:
			{
				VulkanTensor& src = buffers[record.post_download.src];
				Tensor& dst = download_post[record.post_download.dst];
				memcpy(dst.data, src.mapped_data(), src.total() * src.depth * src.packing);
				break;
			}
			default:
				LOG(FATAL) << "invalid record type";
				break;
			}
		}

		delayed_records.clear();
	}

	void ComputeCommand::Reset()
	{
		delayed_records.clear();

		buffers.clear();
		download_post.clear();

		VkResult ret;
		ret = vkResetCommandBuffer(command_buffer, 0);
		CHECK_EQ(VK_SUCCESS, ret);

		ret = vkResetFences(vkdev->GetDevice(), 1, &fence);
		CHECK_EQ(VK_SUCCESS, ret);

		VkCommandBufferBeginInfo command_buffer_begin_info{};
		command_buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		command_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		ret = vkBeginCommandBuffer(command_buffer, &command_buffer_begin_info);
		CHECK_EQ(VK_SUCCESS, ret);
	}


	TransferCommand::TransferCommand(const VulkanDevice* vkdev) : Command(vkdev)
	{
		VkResult ret;
		VkCommandPoolCreateInfo pool_info{};
		pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		pool_info.queueFamilyIndex = vkdev->info.transfer_queue_family_index;
		pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

		ret = vkCreateCommandPool(vkdev->GetDevice(), &pool_info, nullptr, &command_pool);
		CHECK_EQ(VK_SUCCESS, ret) << "vkCreateCommandPool failed " << ret;

		VkCommandBufferAllocateInfo command_buffer_allocate_info{};
		command_buffer_allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		command_buffer_allocate_info.commandPool = command_pool;
		command_buffer_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		command_buffer_allocate_info.commandBufferCount = 1;
		ret = vkAllocateCommandBuffers(vkdev->GetDevice(), &command_buffer_allocate_info, &command_buffer);
		CHECK_EQ(VK_SUCCESS, ret);

		VkFenceCreateInfo fence_create_info{};
		fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		ret = vkCreateFence(vkdev->GetDevice(), &fence_create_info, 0, &fence);
		CHECK_EQ(VK_SUCCESS, ret);
	}

	TransferCommand::~TransferCommand()
	{
		vkDestroyFence(vkdev->GetDevice(), fence, nullptr);
		vkDestroyCommandPool(vkdev->GetDevice(), command_pool, nullptr);
	}

	void TransferCommand::RecordUpload(const Tensor& src, VulkanTensor& dst, const Option& opt)
	{
		if (dst.empty()) dst.CreateLike(src, opt.blob_vkallocator);
		if (dst.allocator->mappable)
		{
			memcpy(dst.mapped_data(), src.data, src.total() * src.depth * src.packing);
		}
		else
		{
			VulkanTensor staging;
			staging.CreateLike(src, opt.staging_vkallocator);

			memcpy(staging.mapped_data(), src.data, src.total() * src.depth * src.packing);

			VkCommandBufferBeginInfo begin_info{};
			begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			vkBeginCommandBuffer(command_buffer, &begin_info);

			dst.data->access_flag = VK_ACCESS_TRANSFER_WRITE_BIT;
			dst.data->stage_flag = VK_PIPELINE_STAGE_TRANSFER_BIT;
			
			VkBufferCopy copy_region{};
			copy_region.size = staging.data->capacity;
			copy_region.dstOffset = dst.data->offset;
			copy_region.srcOffset = staging.data->offset;
			vkCmdCopyBuffer(command_buffer, staging.data->buffer, dst.data->buffer, 1, &copy_region);

			vkEndCommandBuffer(command_buffer);

			VkSubmitInfo submit_info{};
			submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submit_info.commandBufferCount = 1;
			submit_info.pCommandBuffers = &command_buffer;

			VkQueue transfer_queue = vkdev->AcquireQueue(vkdev->info.transfer_queue_family_index);

			vkQueueSubmit(transfer_queue, 1, &submit_info, VK_NULL_HANDLE);
			vkQueueWaitIdle(transfer_queue);

			vkdev->ReclaimQueue(vkdev->info.transfer_queue_family_index, transfer_queue);
		}
	}



	constexpr uint32 MAX_FRAMES_IN_FLIGHT = 2;
	GraphicsCommand::GraphicsCommand(const VulkanDevice* vkdev) : Command(vkdev)
	{
		VkCommandPoolCreateInfo pool_info{};
		pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		pool_info.queueFamilyIndex = vkdev->info.graphics_queue_family_index;
		pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

		VkResult ret = vkCreateCommandPool(vkdev->GetDevice(), &pool_info, nullptr, &command_pool);
		CHECK_EQ(VK_SUCCESS, ret) << "vkCreateCommandPool failed " << ret;
	}

	GraphicsCommand::~GraphicsCommand()
	{
		vkDestroyDescriptorPool(vkdev->GetDevice(), descriptor_pool, nullptr);

		for (auto& semaphore : render_finished_semaphores)
		{
			vkDestroySemaphore(vkdev->GetDevice(), semaphore, nullptr);
		}
		for (auto& semaphore : image_available_semaphores)
		{
			vkDestroySemaphore(vkdev->GetDevice(), semaphore, nullptr);
		}
		for (auto& fence : in_flight_fences)
		{
			vkDestroyFence(vkdev->GetDevice(), fence, nullptr);
		}

		vkDestroyCommandPool(vkdev->GetDevice(), command_pool, nullptr);
	}


	void GraphicsCommand::Create(size_t buffers_count)
	{
		VkResult ret;

		command_buffers.resize(buffers_count);

		VkCommandBufferAllocateInfo alloc_info{};
		alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		alloc_info.commandPool = command_pool;
		alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		alloc_info.commandBufferCount = (uint32)buffers_count; // (uint32_t)command_buffers.size();

		ret = vkAllocateCommandBuffers(vkdev->GetDevice(), &alloc_info, command_buffers.data());
		CHECK_EQ(VK_SUCCESS, ret) << "vkAllocateCommandBuffers failed " << ret;

		// to create sync objects
		image_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
		render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
		in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);
		images_in_flight.resize(buffers_count, VK_NULL_HANDLE);

		VkSemaphoreCreateInfo semaphore_info{};
		semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fence_info{};
		fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			ret = vkCreateSemaphore(vkdev->GetDevice(), &semaphore_info, nullptr, &image_available_semaphores[i]);
			CHECK_EQ(VK_SUCCESS, ret) << "vkCreateSemaphore failed " << ret;
			ret = vkCreateSemaphore(vkdev->GetDevice(), &semaphore_info, nullptr, &render_finished_semaphores[i]);
			CHECK_EQ(VK_SUCCESS, ret) << "vkCreateSemaphore failed " << ret;
			ret = vkCreateFence(vkdev->GetDevice(), &fence_info, nullptr, &in_flight_fences[i]);
			CHECK_EQ(VK_SUCCESS, ret) << "vkCreateFence failed " << ret;
		}
	}
	void GraphicsCommand::RecordUpload(const Tensor& src, VulkanTensor& dst, VulkanAllocator* allocator, VulkanAllocator* staging_allocator)
	{
		if (dst.empty()) dst.CreateLike(src, allocator);
		if (dst.allocator->mappable)
		{
			memcpy(dst.mapped_data(), src.data, src.total() * src.depth * src.packing);
		}
		else
		{
			VulkanTensor staging;
			staging.CreateLike(src, staging_allocator);

			memcpy(staging.mapped_data(), src.data, src.total() * src.depth * src.packing);

			VkCommandBufferBeginInfo begin_info{};
			begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			vkBeginCommandBuffer(command_buffers[0], &begin_info);

			dst.data->access_flag = VK_ACCESS_TRANSFER_WRITE_BIT;
			dst.data->stage_flag = VK_PIPELINE_STAGE_TRANSFER_BIT;

			VkBufferCopy copy_region{};
			copy_region.size = staging.data->capacity;
			copy_region.dstOffset = dst.data->offset;
			copy_region.srcOffset = staging.data->offset;
			vkCmdCopyBuffer(command_buffers[0], staging.data->buffer, dst.data->buffer, 1, &copy_region);

			vkEndCommandBuffer(command_buffers[0]);

			VkSubmitInfo submit_info{};
			submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submit_info.commandBufferCount = 1;
			submit_info.pCommandBuffers = &command_buffers[0];

			VkQueue transfer_queue = vkdev->AcquireQueue(vkdev->info.transfer_queue_family_index);

			vkQueueSubmit(transfer_queue, 1, &submit_info, VK_NULL_HANDLE);
			vkQueueWaitIdle(transfer_queue);

			vkdev->ReclaimQueue(vkdev->info.transfer_queue_family_index, transfer_queue);
		}
	}

	void GraphicsCommand::RecordPipeline(const GraphicsPipeline* pipeline, uint32 buffers_count, VkFramebuffer* frame_buffers,
		VkExtent2D extent, const VulkanTensor& vertex, const VulkanTensor& indices, const std::vector<VulkanTensor>& uniform)
	{
		VkResult ret;

		VkDescriptorPoolSize pool_size{};
		pool_size.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		pool_size.descriptorCount = buffers_count;

		VkDescriptorPoolCreateInfo pool_info{};
		pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		pool_info.poolSizeCount = 1;
		pool_info.pPoolSizes = &pool_size;
		pool_info.maxSets = buffers_count;

		ret = vkCreateDescriptorPool(vkdev->GetDevice(), &pool_info, nullptr, &descriptor_pool);
		CHECK_EQ(VK_SUCCESS, ret) << "vkCreateDescriptorPool failed " << ret;

		std::vector<VkDescriptorSetLayout> layouts(buffers_count, pipeline->descriptorset_layout);
		VkDescriptorSetAllocateInfo alloc_info{};
		alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		alloc_info.descriptorPool = descriptor_pool;
		alloc_info.descriptorSetCount = buffers_count;
		alloc_info.pSetLayouts = layouts.data();

		descriptorsets.resize(buffers_count);
		ret = vkAllocateDescriptorSets(vkdev->GetDevice(), &alloc_info, descriptorsets.data());
		CHECK_EQ(VK_SUCCESS, ret) << "vkAllocateDescriptorSets failed " << ret;

		for (uint32_t i = 0; i < buffers_count; i++)
		{
			VkDescriptorBufferInfo buffer_info{};
			buffer_info.buffer = uniform[i].data->buffer;
			buffer_info.offset = uniform[i].data->offset;
			buffer_info.range = uniform[i].data->capacity;

			VkWriteDescriptorSet descriptor_write{};
			descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptor_write.dstSet = descriptorsets[i];
			descriptor_write.dstBinding = 0;
			descriptor_write.dstArrayElement = 0;
			descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptor_write.descriptorCount = 1;
			descriptor_write.pBufferInfo = &buffer_info;

			vkUpdateDescriptorSets(vkdev->GetDevice(), 1, &descriptor_write, 0, nullptr);

			VkCommandBufferBeginInfo begin_info{};
			begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			ret = vkBeginCommandBuffer(command_buffers[i], &begin_info);
			CHECK_EQ(VK_SUCCESS, ret) << "vkBeginCommandBuffer failed " << ret;

			VkRenderPassBeginInfo render_pass_info{};
			render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			render_pass_info.renderPass = pipeline->render_pass;
			render_pass_info.framebuffer = frame_buffers[i];
			render_pass_info.renderArea.offset = { 0, 0 };
			render_pass_info.renderArea.extent = extent;

			VkClearValue clear_color = { 0.0f, 0.0f, 0.0f, 1.0f };
			render_pass_info.clearValueCount = 1;
			render_pass_info.pClearValues = &clear_color;

			vkCmdBeginRenderPass(command_buffers[i], &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
			{
				vkCmdBindPipeline(command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->pipeline);

				VkBuffer vertex_buffers[] = { vertex.data->buffer };
				VkDeviceSize offsets[] = { vertex.data->offset }; // ?
				vkCmdBindVertexBuffers(command_buffers[i], 0, 1, vertex_buffers, offsets);
				vkCmdBindIndexBuffer(command_buffers[i], indices.data->buffer, indices.data->offset, VK_INDEX_TYPE_UINT16); // dtype -> VK_INDEX_TYPE_UINT16 ?

				vkCmdBindDescriptorSets(command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->pipeline_layout, 0, 1, &descriptorsets[i], 0, nullptr);

				//vkCmdDraw(command_buffers[i], 3, 1, 0, 0);
				vkCmdDrawIndexed(command_buffers[i], static_cast<uint32>(indices.shape.total()), 1, 0, 0, 0);
			}
			vkCmdEndRenderPass(command_buffers[i]);

			ret = vkEndCommandBuffer(command_buffers[i]);
			CHECK_EQ(VK_SUCCESS, ret) << "vkEndCommandBuffer failed " << ret;
		}
	}

	void GraphicsCommand::Present(VkSwapchainKHR swap_chain, uint32 present_queue_family_index, const std::function<void(uint32)>& UpdateUniformBuffer)
	{
		VkResult ret;
		vkWaitForFences(vkdev->GetDevice(), 1, &in_flight_fences[current_frame], VK_TRUE, UINT64_MAX);

		uint32_t image_index;
		ret = vkAcquireNextImageKHR(vkdev->GetDevice(), swap_chain, UINT64_MAX, image_available_semaphores[current_frame], VK_NULL_HANDLE, &image_index);
		//if (ret == VK_ERROR_OUT_OF_DATE_KHR)
		//{
		//	RecreateSwapChain();
		//	return;
		//}
		CHECK(ret == VK_SUCCESS || ret == VK_SUBOPTIMAL_KHR) << "failed to acquire swap chain image";

		UpdateUniformBuffer(image_index);

		if (images_in_flight[image_index] != VK_NULL_HANDLE)
		{
			vkWaitForFences(vkdev->GetDevice(), 1, &images_in_flight[image_index], VK_TRUE, UINT64_MAX);
		}
		images_in_flight[image_index] = in_flight_fences[current_frame];

		VkSubmitInfo submit_info{};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore waitSemaphores[] = { image_available_semaphores[current_frame] };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submit_info.waitSemaphoreCount = 1;
		submit_info.pWaitSemaphores = waitSemaphores;
		submit_info.pWaitDstStageMask = waitStages;

		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffers[image_index];

		VkSemaphore signalSemaphores[] = { render_finished_semaphores[current_frame] };
		submit_info.signalSemaphoreCount = 1;
		submit_info.pSignalSemaphores = signalSemaphores;

		vkResetFences(vkdev->GetDevice(), 1, &in_flight_fences[current_frame]);

		VkQueue graphics_queue = vkdev->AcquireQueue(vkdev->info.graphics_queue_family_index);

		ret = vkQueueSubmit(graphics_queue, 1, &submit_info, in_flight_fences[current_frame]);
		CHECK_EQ(VK_SUCCESS, ret) << "vkQueueSubmit failed " << ret;

		vkdev->ReclaimQueue(vkdev->info.graphics_queue_family_index, graphics_queue);

		VkPresentInfoKHR present_info{};
		present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		present_info.waitSemaphoreCount = 1;
		present_info.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swap_chains[] = { swap_chain };
		present_info.swapchainCount = 1;
		present_info.pSwapchains = swap_chains;

		present_info.pImageIndices = &image_index;


		VkQueue present_queue = vkdev->AcquireQueue(present_queue_family_index);
		ret = vkQueuePresentKHR(present_queue, &present_info);
		vkdev->ReclaimQueue(present_queue_family_index, present_queue);
		//vkdev->ReclaimQueue(vkdev->info.graphics_queue_family_index, graphics_queue);

		//if (ret == VK_ERROR_OUT_OF_DATE_KHR || ret == VK_SUBOPTIMAL_KHR || frame_buffer_resized)
		//{
		//	frame_buffer_resized = false;
		//	RecreateSwapChain();
		//}
		//else
		//{
		//	CHECK_EQ(VK_SUCCESS, ret) << "vkQueuePresentKHR failed " << ret;
		//}

		current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
	}
}