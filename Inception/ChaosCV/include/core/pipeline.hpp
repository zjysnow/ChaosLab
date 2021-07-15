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

		void CreateDescriptorSetLayout(const size_t& binding_count, const VkShaderStageFlagBits* flags, const VkDescriptorType* types);
		void CreatePipelineLayout(size_t push_constant_count = 0);
		VkShaderModule CompileShaderModule(const uint32* spv_data, size_t spv_data_size);

		int layout_flags;
	};

	class CHAOS_API ComputePipeline : public Pipeline
	{
	public:
		ComputePipeline(const VulkanDevice* vkdev);
		~ComputePipeline();

		void Create(const uint32* comp_data, size_t comp_size, const char* entry_name, const std::vector<VulkanSpecializationType>& specializations, size_t binding_count, size_t push_constant_count);

		void SetOptimalLocalSize(uint32 x, uint32 y, uint32 z);

		VkDescriptorUpdateTemplate descriptor_update_template;

		uint32 local_size_x = 1;
		uint32 local_size_y = 1;
		uint32 local_size_z = 1;
	private:
		void CreateDescriptorUpdateTemplate(size_t binding_count);

		VkShaderModule comp;
	};

	class CHAOS_API GraphicsPipeline : public Pipeline
	{
	public:
		GraphicsPipeline(const VulkanDevice* vkdev);
		~GraphicsPipeline();

		void Create(const uint32* vert_data, size_t vert_size, const uint32* frag_data, size_t frag_size,
			VkExtent2D extent, VkFormat format, VkPolygonMode polygon_mode, VkFrontFace front_face, VkPrimitiveTopology topoloty, VkCullModeFlagBits cull_mode);

		void CreateRenderPass(const VkFormat& format);

		VkRenderPass render_pass;

		VkShaderModule vert;
		VkShaderModule frag;
	};
}