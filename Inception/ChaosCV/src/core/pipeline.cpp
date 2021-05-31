#include "core/pipeline.hpp"

#include <vulkan/vulkan.hpp>

namespace chaos
{
	Pipeline::Pipeline(const VulkanDevice* vkdev) : vkdev(vkdev) {}
	Pipeline::~Pipeline()
	{
		vkDestroyDescriptorSetLayout(vkdev->GetDevice(), descriptorset_layout, nullptr);
		vkDestroyPipeline(vkdev->GetDevice(), pipeline, nullptr);
		vkDestroyPipelineLayout(vkdev->GetDevice(), pipeline_layout, nullptr);
	}

	void Pipeline::CreateDescriptorSetLayout(const uint32& binding_count, const ShaderStageFlag* flags, const DescriptorType* types)
	{
		std::vector<VkDescriptorSetLayoutBinding> bingdings(binding_count);
		for (uint32 i = 0; i < binding_count; i++)
		{
			bingdings[i].binding = i;
			bingdings[i].descriptorCount = 1;
			bingdings[i].descriptorType = static_cast<VkDescriptorType>(types[i]); //VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			bingdings[i].pImmutableSamplers = nullptr; // immutable texelfetch sampler
			bingdings[i].stageFlags = flags[i]; //VK_SHADER_STAGE_VERTEX_BIT;
		}

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = binding_count;
		layoutInfo.pBindings = bingdings.data();

		auto ret = vkCreateDescriptorSetLayout(vkdev->GetDevice(), &layoutInfo, nullptr, &descriptorset_layout);
		CHECK_EQ(VK_SUCCESS, ret) << "vkCreateDescriptorSetLayout failed " << ret;
	}


	void Pipeline::CreatePipelineLayout(int push_constant_count)
	{
		VkPipelineLayoutCreateInfo pipeline_layout_info{};
		pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		if (descriptorset_layout)
		{
			pipeline_layout_info.setLayoutCount = 1;
			pipeline_layout_info.pSetLayouts = &descriptorset_layout;
		}

		if (push_constant_count > 0)
		{
			VkPushConstantRange push_constant_range{};
			push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			push_constant_range.size = sizeof(VulkanConstantType) * push_constant_count;

			pipeline_layout_info.pushConstantRangeCount = 1;
			pipeline_layout_info.pPushConstantRanges = &push_constant_range;
		}

		VkResult ret = vkCreatePipelineLayout(vkdev->GetDevice(), &pipeline_layout_info, nullptr, &pipeline_layout);
		CHECK_EQ(VK_SUCCESS, ret) << "vkCreatePipelineLayout failed " << ret;
	}
	VkShaderModule Pipeline::CompileShaderModule(const uint32* spv_data, size_t spv_data_size)
	{
		VkShaderModuleCreateInfo shader_module_create_info{};
		shader_module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		shader_module_create_info.codeSize = spv_data_size;
		shader_module_create_info.pCode = spv_data;

		VkShaderModule shader_module{};
		VkResult ret = vkCreateShaderModule(vkdev->GetDevice(), &shader_module_create_info, nullptr, &shader_module);
		CHECK_EQ(VK_SUCCESS, ret) << "vkCreateShaderModule failed " << ret;

		return shader_module;
	}

}