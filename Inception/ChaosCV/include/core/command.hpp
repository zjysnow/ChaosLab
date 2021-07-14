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
		~ComputeCommand();

		void RecordUpload(const Tensor& src, VulkanTensor& dst, const Option& opt);
		void RecordDownload(const VulkanTensor& src, Tensor& dst, const Option& opt);
		void RecordClone(const VulkanTensor& src, VulkanTensor& dst, const Option& opt);

		void RecordPipeline(const ComputePipeline* pipeline, const std::vector<VulkanTensor>& buffer_bindings, const std::vector<VulkanConstantType>& constants, const Shape& dispatcher);

		void SubmitAndWait();
		void Reset();

		void Release();
	protected:
		VkCommandBuffer command_buffer;
		VkFence fence;

		class Record
		{
		public:
			enum
			{
				TYPE_DOWNLOAD,
			};
			int type;

			union
			{
				struct
				{
					uint32 src;
					uint32 dst;
				} post_download;
			};
		};

		std::vector<Record> delayed_records;

		std::vector<Tensor> download_post;
		std::vector<VulkanTensor> buffers; // to buffers and record pipeline bindings buffer too
		
		std::function<void(VkCommandBuffer, VkDescriptorUpdateTemplate, VkPipelineLayout, uint32, const void*)> vkCmdPushDescriptorSetWithTemplate;
	};

	class CHAOS_API TransferCommand : public Command
	{
	public:
		TransferCommand(const VulkanDevice* vkdev);
		~TransferCommand();

		void RecordUpload(const Tensor& src, VulkanTensor& dst, const Option& opt);
		
	private:
		VkCommandBuffer command_buffer;
		VkFence fence;
	};

	class GraphicsPipeline;
	class CHAOS_API GraphicsCommand : public Command
	{
	public:
		GraphicsCommand(const VulkanDevice* vkdev);
		~GraphicsCommand();

		void RecordUpload(const Tensor& src, VulkanTensor& dst, VulkanAllocator* allocator, VulkanAllocator* staging_allocator);

		void Create(size_t buffers_count);

		void RecordPipeline(const GraphicsPipeline* pipeline, uint32 buffers_count, VkFramebuffer* frame_buffers,
			VkExtent2D extent, const VulkanTensor& vertex, const VulkanTensor& indices, const std::vector<VulkanTensor>& uniform);

		void Present(VkSwapchainKHR swap_chain, uint32 present_queue_family_index, const std::function<void(uint32)>& UpdateUniformBuffer);

	protected:
		std::vector<VkCommandBuffer> command_buffers;

		std::vector<VkSemaphore> image_available_semaphores;
		std::vector<VkSemaphore> render_finished_semaphores;
		std::vector<VkFence> in_flight_fences;
		std::vector<VkFence> images_in_flight;

		VkDescriptorPool descriptor_pool;
		std::vector<VkDescriptorSet> descriptorsets;

		size_t current_frame = 0;
	};
}