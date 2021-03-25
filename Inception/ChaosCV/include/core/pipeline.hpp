#pragma once

#include "core/core.hpp"

namespace chaos
{
	class CHAOS_API GraphicsPipeline
	{
	public:
		GraphicsPipeline(const VulkanDevice* vkdev);
		~GraphicsPipeline();

		void CreateShaderModule();

		const VulkanDevice* vkdev;

		VkPipelineLayout pipeline_layout;
		VkPipeline pipeline;

		VkDescriptorSetLayout descriptorset_layout;

		VkShaderModule vert;
		VkShaderModule frag;
	};
}