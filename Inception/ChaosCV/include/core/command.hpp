#pragma once

#include "core/core.hpp"
#include "core/gpu.hpp"
#include "core/buffer.hpp"
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
		~GraphicsCommand();

		void RecordClone(const VulkanTensor& src, VulkanTensor& dst, VulkanAllocator* allocator);

		//void CreateSyncObjects(size_t count);
		void Init(size_t buffers_count);
		void FreeCommandBuffers();

		void RecordPipeline(const GraphicsPipeline* pipeline, uint32 buffers_count, VkFramebuffer* frame_buffers, 
			uint32 width, uint32 height, const VulkanTensor& vertex, const VulkanTensor& indices, const std::vector<VulkanTensor>& uniform);
		void Present(VkSwapchainKHR swap_chain, uint32 present_queue_family_index);
	protected:
		std::vector<VkCommandBuffer> command_buffers;

		std::vector<VkSemaphore> image_available_semaphores;
		std::vector<VkSemaphore> render_finished_semaphores;
		std::vector<VkFence> in_flight_fences;
		std::vector<VkFence> images_in_flight;

		//VkDescriptorPool descriptor_pool;
		//std::vector<VkDescriptorSet> descriptorsets;

		size_t current_frame = 0;
	};

}