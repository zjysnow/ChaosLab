#pragma once

#include "core/core.hpp"
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

		void CreateDescriptorSetLayout(const uint32& binding_count, const VkShaderStageFlagBits* flags, const VkDescriptorType* types);
		void CreatePipelineLayout(int push_constant_count = 0);
		VkShaderModule CompileShaderModule(const uint32* spv_data, size_t spv_data_size);
	};

	class CHAOS_API ComputePipeline : public Pipeline
	{
	public:
		ComputePipeline(const VulkanDevice* vkdev);
		~ComputePipeline();

		VkShaderModule comp;
	};

	class CHAOS_API GraphicsPipeline : public Pipeline
	{
	public:
		GraphicsPipeline(const VulkanDevice* vkdev);
		~GraphicsPipeline();

		void Create(const uint32* vert_data, size_t vert_size, const uint32* frag_data, size_t frag_size, 
			VkExtent2D extent, VkFormat format, VkPolygonMode polygon_mode, VkFrontFace front_face, VkPrimitiveTopology topoloty, VkCullModeFlagBits cull_mode);
		
		void CreateRenderPass(VkFormat format);

		VkRenderPass render_pass;

		VkDescriptorType descriptor_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		VkShaderStageFlagBits shader_stage_flag = VK_SHADER_STAGE_VERTEX_BIT;

		VkShaderModule vert;
		VkShaderModule frag;
	};
}