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
	}
	ComputeCommand::~ComputeCommand()
	{
		

		vkDestroyFence(vkdev->GetDevice(), fence, nullptr);

		vkDestroyCommandPool(vkdev->GetDevice(), command_pool, nullptr);
		
	}

	void ComputeCommand::RecordUpload(const Tensor& src, VulkanTensor& dst, VulkanAllocator* allocator)
	{
	
		dst.CreateLike(src, allocator);
		if (allocator->mappable)
		{
			memcpy(dst.mapped_data(), src.data, src.total() * src.depth * src.packing);
		}
		else
		{
			VkResult ret;
			VkCommandBufferBeginInfo command_buffer_begin_info{};
			command_buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			command_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			ret = vkBeginCommandBuffer(command_buffer, &command_buffer_begin_info);
			CHECK_EQ(VK_SUCCESS, ret);

			VulkanTensor staging;
			VulkanAllocator* staging_vkallocator = new VulkanStagingAllocator(vkdev);
			staging.CreateLike(src, staging_vkallocator);
			memcpy(staging.mapped_data(), src.data, src.total() * src.depth * src.packing);

			VkBufferCopy copy_region{};
			copy_region.size = staging.data->capacity;
			copy_region.dstOffset = dst.buffer_offset();
			copy_region.srcOffset = staging.buffer_offset();
			vkCmdCopyBuffer(command_buffer, staging.buffer(), dst.buffer(), 1, &copy_region);

			ret = vkEndCommandBuffer(command_buffer);
			CHECK_EQ(VK_SUCCESS, ret);

			VkSubmitInfo submit_info{};
			submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submit_info.commandBufferCount = 1;
			submit_info.pCommandBuffers = &command_buffer;

			VkQueue compute_queue = vkdev->AcquireQueue(vkdev->info.compute_queue_family_index);

			vkQueueSubmit(compute_queue, 1, &submit_info, VK_NULL_HANDLE);
			vkQueueWaitIdle(compute_queue);

			vkdev->ReclaimQueue(vkdev->info.compute_queue_family_index, compute_queue);

			staging.Release();
			delete staging_vkallocator;

		}
		

		
	}
	void ComputeCommand::RecordDownload(const VulkanTensor& src, Tensor& dst, Allocator* allocator)
	{
		

		dst.CreateLike(src, allocator);
		if (src.allocator->mappable)
		{
			memcpy(dst.data, src.mapped_data(), src.total() * src.depth * src.packing);
		}
		else
		{
			VkResult ret;
			VkCommandBufferBeginInfo command_buffer_begin_info{};
			command_buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			command_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			ret = vkBeginCommandBuffer(command_buffer, &command_buffer_begin_info);
			CHECK_EQ(VK_SUCCESS, ret);

			VulkanTensor staging;
			VulkanAllocator* staging_vkallocator = new VulkanStagingAllocator(vkdev);
			staging.CreateLike(src, staging_vkallocator);

			VkBufferCopy copy_region{};
			copy_region.size = src.data->capacity;
			copy_region.dstOffset = staging.buffer_offset();
			copy_region.srcOffset = src.buffer_offset();
			vkCmdCopyBuffer(command_buffer, src.buffer(), staging.buffer(), 1, &copy_region);

			ret = vkEndCommandBuffer(command_buffer);
			CHECK_EQ(VK_SUCCESS, ret);

			VkSubmitInfo submit_info{};
			submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submit_info.commandBufferCount = 1;
			submit_info.pCommandBuffers = &command_buffer;

			VkQueue compute_queue = vkdev->AcquireQueue(vkdev->info.compute_queue_family_index);

			vkQueueSubmit(compute_queue, 1, &submit_info, VK_NULL_HANDLE);
			vkQueueWaitIdle(compute_queue);

			vkdev->ReclaimQueue(vkdev->info.compute_queue_family_index, compute_queue);

			memcpy(dst.data, staging.mapped_data(), staging.total() * staging.depth * staging.packing);

			staging.Release();
			delete staging_vkallocator;
		}
		
	}

	

	void ComputeCommand::RecordPipeline(const ComputePipeline* pipeline)
	{
		// record pipeline
		//vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);

	}

	void ComputeCommand::SubmitAndWait()
	{
		
		VkResult ret;
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
	}

}