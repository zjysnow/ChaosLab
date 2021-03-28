#include "core/command.hpp"
#include "core/pipeline.hpp"

namespace chaos
{
	Command::Command(const VulkanDevice* vkdev) : vkdev(vkdev)
	{
		VkCommandPoolCreateInfo pool_info{};
		pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		pool_info.queueFamilyIndex = vkdev->info.graphics_queue_family_index;

		VkResult ret = vkCreateCommandPool(vkdev->GetDevice(), &pool_info, nullptr, &command_pool);
		CHECK_EQ(VK_SUCCESS, ret) << "vkCreateCommandPool failed " << ret;
	}
	Command::~Command()
	{
		vkDestroyCommandPool(vkdev->GetDevice(), command_pool, nullptr);
	}

	void Command::RecordUpload(const Tensor& src, VulkanTensor& dst, VulkanAllocator* allocator)
	{
		CHECK(allocator->mappable) << "can not upload to a local memory";
		dst.CreateLike(src, allocator);
		memcpy(dst.mapped_data(), src.data, src.total() * src.dtype * src.packing);
	}


	constexpr uint32 MAX_FRAMES_IN_FLIGHT = 2;

	GraphicsCommand::GraphicsCommand(const VulkanDevice* vkdev) : Command(vkdev)
	{
		
	}

	void GraphicsCommand::Init(size_t buffers_count)
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

	void GraphicsCommand::RecordClone(const VulkanTensor& src, VulkanTensor& dst, VulkanAllocator* allocator)
	{
		VkCommandBufferBeginInfo begin_info{};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		dst.CreateLike(src, allocator);

		vkBeginCommandBuffer(command_buffers[0], &begin_info);
		VkBufferCopy copy_region{};
		copy_region.size = std::min(src.data->capacity, dst.data->capacity);
		copy_region.dstOffset = dst.data->offset;
		copy_region.srcOffset = src.data->offset;
		vkCmdCopyBuffer(command_buffers[0], src.data->buffer, dst.data->buffer, 1, &copy_region);
		vkEndCommandBuffer(command_buffers[0]);


		VkSubmitInfo submit_info{};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffers[0];

		VkQueue graphics_queue = vkdev->AcquireQueue(vkdev->info.graphics_queue_family_index);

		vkQueueSubmit(graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphics_queue);

		vkdev->ReclaimQueue(vkdev->info.graphics_queue_family_index, graphics_queue);

		vkFreeCommandBuffers(vkdev->GetDevice(), command_pool, 1, &command_buffers[0]);
	}

	void GraphicsCommand::RecordPipeline(const GraphicsPipeline* pipeline, const std::vector<VkFramebuffer>& frame_buffers, const VkExtent2D& extent, const VulkanTensor& vertex, const VulkanTensor& indices, const std::vector<VulkanTensor>& uniform)
	{
		VkResult ret;
		uint32 buffers_count = static_cast<uint32>(frame_buffers.size());

		VkDescriptorPoolSize pool_size{};
		pool_size.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		pool_size.descriptorCount = buffers_count; // static_cast<uint32_t>(swapChainImages.size());

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = 1;
		poolInfo.pPoolSizes = &pool_size;
		poolInfo.maxSets = buffers_count; // static_cast<uint32_t>(swapChainImages.size());

		ret = vkCreateDescriptorPool(vkdev->GetDevice(), &poolInfo, nullptr, &descriptor_pool);

		std::vector<VkDescriptorSetLayout> layouts(buffers_count, pipeline->descriptorset_layout);
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptor_pool;
		allocInfo.descriptorSetCount = buffers_count; //static_cast<uint32_t>(swapChainImages.size());
		allocInfo.pSetLayouts = layouts.data();

		descriptorsets.resize(buffers_count);
		ret = vkAllocateDescriptorSets(vkdev->GetDevice(), &allocInfo, descriptorsets.data());
		CHECK_EQ(VK_SUCCESS, ret) << "vkAllocateDescriptorSets failed " << ret;


		for (uint32_t i = 0; i < buffers_count; i++)
		{
			VkDescriptorBufferInfo buffer_info{};
			buffer_info.buffer = uniform[i].data->buffer;
			buffer_info.offset = uniform[i].data->offset;
			buffer_info.range = uniform[i].data->capacity;

			VkWriteDescriptorSet descriptorWrite{};
			descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrite.dstSet = descriptorsets[i];
			descriptorWrite.dstBinding = 0;
			descriptorWrite.dstArrayElement = 0;
			descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrite.descriptorCount = 1;
			descriptorWrite.pBufferInfo = &buffer_info;

			vkUpdateDescriptorSets(vkdev->GetDevice(), 1, &descriptorWrite, 0, nullptr);

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
}