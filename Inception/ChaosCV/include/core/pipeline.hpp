#pragma once

#include "core/gpu.hpp"

namespace chaos
{
	class CHAOS_API Pipeline
	{
	public:
		virtual ~Pipeline();

		const VulkanDevice* vkdev;

		VkPipeline pipeline;
		VkPipelineLayout pipeline_layout;

		VkDescriptorSetLayout descriptorset_layout;

	protected:
		Pipeline(const VulkanDevice* vkdev);

		void CreateDescriptorSetLayout(const uint32& binding_count, const ShaderStageFlag* flags, const DescriptorType* types);
		void CreatePipelineLayout(int push_constant_count = 0);
		VkShaderModule CompileShaderModule(const uint32* spv_data, size_t spv_data_size);
	};

	class CHAOS_API ComputePipeline : public Pipeline
	{
	public:
		ComputePipeline(const VulkanDevice* vkdev);



		VkShaderModule comp;
	};
}