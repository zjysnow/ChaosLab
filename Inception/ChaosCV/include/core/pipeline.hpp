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

		void CreateDescriptorSetLayout(const size_t& binding_count, const ShaderStageFlag* flags, const DescriptorType* types);
		void CreatePipelineLayout(size_t push_constant_count = 0);
		VkShaderModule CompileShaderModule(const uint32* spv_data, size_t spv_data_size);
	};

	class CHAOS_API ComputePipeline : public Pipeline
	{
	public:
		ComputePipeline(const VulkanDevice* vkdev);
		~ComputePipeline();

		void Create(const uint32* comp_data, size_t comp_size, const char* entry_name, const std::vector<VulkanSpecializationType>& specializations, size_t binding_count, size_t push_constant_count);

		void SetOptimalLocalSize(uint32 x, uint32 y, uint32 z);

		void CreateDescriptorUpdateTemplate(size_t binding_count);

		std::vector<DescriptorType> descriptor_types; // = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		std::vector<ShaderStageFlag> shader_stage_flags; // = VK_SHADER_STAGE_VERTEX_BIT;

		uint32 local_size_x = 1;
		uint32 local_size_y = 1;
		uint32 local_size_z = 1;
		VkShaderModule comp;
		VkDescriptorUpdateTemplate descriptor_update_template;
	};
}