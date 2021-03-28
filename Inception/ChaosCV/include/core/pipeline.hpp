#pragma once

#include "core/core.hpp"
#include "core/gpu.hpp"

namespace chaos
{
	class CHAOS_API Pipeline
	{
	public:
		Pipeline(const VulkanDevice* vkdev);
		virtual ~Pipeline();

		const VulkanDevice* vkdev;

		VkPipeline pipeline;
		VkPipelineLayout pipeline_layout;

		VkDescriptorPool discriptor_pool;
		VkDescriptorSetLayout descriptorset_layout;

	protected:
		void CreateDescriptorSetLayout(const uint32& binding_count, const VkShaderStageFlagBits* flags, const VkDescriptorType* types);
		void CreatePipelineLayout();
		VkShaderModule CompileShaderModule(const uint32* spv_data, size_t spv_data_size);
	};

	class CHAOS_API GraphicsPipeline : public Pipeline
	{
	public:
		GraphicsPipeline(const VulkanDevice* vkdev);
		~GraphicsPipeline();

		void Create(const uint32* vert_data, size_t vert_size, const uint32* frag_data, size_t frag_size, 
			const VkFormat& format, const VkExtent2D& extent, const VkPolygonMode& polygon_mode);
		
		void CreateRenderPass(const VkFormat& format);

		const VulkanDevice* vkdev;

		VkPipelineLayout pipeline_layout;
		VkPipeline pipeline;

		VkRenderPass render_pass;

		VkDescriptorPool discriptor_pool;
		std::vector<VkDescriptorSet> descriptor_sets;
		VkDescriptorSetLayout descriptorset_layout;

		VkDescriptorType descriptor_type;
		VkShaderStageFlagBits shader_stage_flag;

		VkShaderModule vert;
		VkShaderModule frag;
	};
}