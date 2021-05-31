#pragma once

#include "core/gpu.hpp"
#include "core/tensor.hpp"

namespace chaos
{
	class Option;
	class CHAOS_API Command
	{
	public:
		Command(const VulkanDevice* vkdev);
		virtual ~Command();

		const VulkanDevice* vkdev;
		VkCommandPool command_pool;
	};

	class ComputePipeline;
	class CHAOS_API ComputeCommand : public Command
	{
	public:
		ComputeCommand(const VulkanDevice* vkdev);
		virtual ~ComputeCommand();

		void RecordUpload(const Tensor& src, VulkanTensor& dst, VulkanAllocator* allocator);
		void RecordDownload(const VulkanTensor& src, Tensor& dst, Allocator* allocator);


		void RecordPipeline(const ComputePipeline* pipeline);

		void SubmitAndWait();
	protected:
		VkCommandBuffer command_buffer;
		VkFence fence;
		VkSemaphore semaphore;
		VkDescriptorPool descriptor_pool;
	};
}