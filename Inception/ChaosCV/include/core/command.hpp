#pragma once

#include "core/core.hpp"
#include "core/gpu.hpp"
#include "core/tensor.hpp"

namespace chaos
{
	class VulkanDevice;
	class CHAOS_API Command
	{
	public:
		Command(const VulkanDevice* vkdev);
		virtual ~Command();

		virtual void RecordUpload(const Tensor& src, VulkanTensor& dst, VulkanAllocator* allocator);
	protected:
		const VulkanDevice* vkdev;
		VkCommandPool command_pool;

	};

	class GraphicsPipeline;
	class CHAOS_API GraphicsCommand : public Command
	{
	public:
		GraphicsCommand(const VulkanDevice* vkdev);
		
		void RecordClone(const VulkanTensor& src, VulkanTensor& dst, VulkanAllocator* allocator);

		//void CreateSyncObjects(size_t count);
		void Init(size_t buffers_count);

		void RecordPipeline(const GraphicsPipeline* pipeline, const std::vector<VkFramebuffer>& frame_buffers, 
			const VkExtent2D& extent, const VulkanTensor& vertex, const VulkanTensor& indices, const std::vector<VulkanTensor>& uniform);
		void Present();
	protected:
		using Command::vkdev;

		std::vector<VkCommandBuffer> command_buffers;

		std::vector<VkSemaphore> image_available_semaphores;
		std::vector<VkSemaphore> render_finished_semaphores;
		std::vector<VkFence> in_flight_fences;
		std::vector<VkFence> images_in_flight;

		VkDescriptorPool descriptor_pool;
		std::vector<VkDescriptorSet> descriptorsets;

		size_t current_frame = 0;
	};

	class CHAOS_API TransferCommand : public Command
	{
	public:
		TransferCommand(const VulkanDevice* vkdev);

		void RecordUpload();

	protected:
		VkCommandBuffer transfer_command_buffer;

		VkFence upload_command_fence;
	};
}