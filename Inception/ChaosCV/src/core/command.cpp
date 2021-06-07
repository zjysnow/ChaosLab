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
		vkDestroyFence(vkdev->GetDevice(), fence, nullptr);
		vkDestroyCommandPool(vkdev->GetDevice(), command_pool, nullptr);
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

			staging_buffers.push_back(staging);
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

			staging_buffers.push_back(staging);
			download_post.push_back(dst);

			Record r;
			r.type = Record::TYPE_DOWNLOAD;
			r.post_download.src = (uint32)staging_buffers.size() - 1;
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
				VulkanTensor& src = staging_buffers[record.post_download.src];
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

		staging_buffers.clear();
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
}