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

		void RecordUpload(const Tensor& src, VulkanTensor& dst, const Option& opt);
		void RecordDownload(const VulkanTensor& src, Tensor& dst, const Option& opt);

		void RecordClone(const VulkanTensor& src, VulkanTensor& dst);
		void RecordClone(const VulkanTensor& src, VulkanTensor& dst, const Option& opt);

		void RecordPipeline(const ComputePipeline* pipeline, const std::vector<VulkanTensor>& buffer_bindings, const std::vector<VulkanConstantType>& constants, const Shape& dispatcher);

		void SubmitAndWait();
	protected:
		VkCommandBuffer command_buffer;
		VkFence fence;
		//VkSemaphore semaphore;
		//VkDescriptorPool descriptor_pool;

		//std::vector<VkDescriptorSet> descriptor_sets;

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
		std::vector<VulkanTensor> buffers;
		
		std::function<void(VkCommandBuffer, VkDescriptorUpdateTemplate, VkPipelineLayout, uint32, const void*)> vkCmdPushDescriptorSetWithTemplate;
	};
}