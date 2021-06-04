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

	void Pipeline::CreateDescriptorSetLayout(const size_t& binding_count, const ShaderStageFlag* flags, const DescriptorType* types)
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
		// VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR
		VkDescriptorSetLayoutCreateInfo layout_info{};
		layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layout_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
		layout_info.bindingCount = (uint32)binding_count;
		layout_info.pBindings = bingdings.data();

		auto ret = vkCreateDescriptorSetLayout(vkdev->GetDevice(), &layout_info, nullptr, &descriptorset_layout);
		CHECK_EQ(VK_SUCCESS, ret) << "vkCreateDescriptorSetLayout failed " << ret;
	}


	void Pipeline::CreatePipelineLayout(size_t push_constant_count)
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
			push_constant_range.size = (uint32)(sizeof(VulkanConstantType) * push_constant_count);

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





	ComputePipeline::ComputePipeline(const VulkanDevice* vkdev) : Pipeline(vkdev) {}
	ComputePipeline::~ComputePipeline()
	{
		if (descriptor_update_template) vkdev->DestroyDescriptorUpdateTemplate(descriptor_update_template);
		vkDestroyShaderModule(vkdev->GetDevice(), comp, nullptr);
	}

	void ComputePipeline::Create(const uint32* comp_data, size_t comp_size, const char* entry_name, const std::vector<VulkanSpecializationType>& specializations, size_t binding_count, size_t push_constant_count)
	{
		descriptor_types = std::vector<DescriptorType>(binding_count, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
		shader_stage_flags = std::vector<ShaderStageFlag>(binding_count, VK_SHADER_STAGE_COMPUTE_BIT);

		comp = CompileShaderModule(comp_data, comp_size);

		CreateDescriptorSetLayout(binding_count, shader_stage_flags.data(), descriptor_types.data());

		CreatePipelineLayout(push_constant_count);

		// create pipeline
		const size_t specialization_count = specializations.size();
		std::vector<VkSpecializationMapEntry> specialization_map_entries(specialization_count);
		for (size_t i = 0; i < specialization_count; i++)
		{
			specialization_map_entries[i].constantID = (uint32)i;
			specialization_map_entries[i].offset = (uint32)(i * sizeof(VulkanSpecializationType));
			specialization_map_entries[i].size = sizeof(VulkanSpecializationType);
		}
		VkSpecializationInfo specialization_info{};
		specialization_info.mapEntryCount = (uint32)specialization_map_entries.size();
		specialization_info.pMapEntries = specialization_map_entries.data();
		specialization_info.dataSize = specializations.size() * sizeof(VulkanSpecializationType);
		specialization_info.pData = specializations.data();

		VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info{};
		pipeline_shader_stage_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		pipeline_shader_stage_create_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		pipeline_shader_stage_create_info.module = comp;
		pipeline_shader_stage_create_info.pName = entry_name; //"main";
		pipeline_shader_stage_create_info.pSpecializationInfo = &specialization_info;

		VkComputePipelineCreateInfo compute_pipeline_create_info{};
		compute_pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		compute_pipeline_create_info.stage = pipeline_shader_stage_create_info;
		compute_pipeline_create_info.layout = pipeline_layout;

		VkResult ret = vkCreateComputePipelines(vkdev->GetDevice(), 0, 1, &compute_pipeline_create_info, 0, &pipeline);
		CHECK_EQ(VK_SUCCESS, ret) << "vkCreateComputePipelines failed " << ret;

		CreateDescriptorUpdateTemplate(binding_count);
	}

	void ComputePipeline::CreateDescriptorUpdateTemplate(size_t binding_count)
	{
		if (binding_count == 0) return;

		std::vector<VkDescriptorUpdateTemplateEntryKHR> descriptor_update_template_entries(binding_count);
		for (size_t i = 0; i < binding_count; i++)// TODO do not update weights
		{
			descriptor_update_template_entries[i].dstBinding = (uint32)i;
			descriptor_update_template_entries[i].dstArrayElement = 0;
			descriptor_update_template_entries[i].descriptorCount = 1;
			descriptor_update_template_entries[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptor_update_template_entries[i].offset = i * sizeof(VkDescriptorBufferInfo);
			descriptor_update_template_entries[i].stride = sizeof(VkDescriptorBufferInfo);
		}

		VkDescriptorUpdateTemplateCreateInfoKHR descriptor_update_template_create_info{};
		descriptor_update_template_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO_KHR;
		descriptor_update_template_create_info.descriptorUpdateEntryCount = (uint32)binding_count;// TODO do not update weights
		descriptor_update_template_create_info.pDescriptorUpdateEntries = descriptor_update_template_entries.data();
		descriptor_update_template_create_info.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR; // ???
		// descriptorSetLayout should be ignored if VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR
		// FIXME HACK WARNING TODO NOTE but crash on radv if set NULL  :(
		descriptor_update_template_create_info.descriptorSetLayout = descriptorset_layout;
		descriptor_update_template_create_info.pipelineBindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
		descriptor_update_template_create_info.pipelineLayout = pipeline_layout;

		//auto vkCreateDescriptorUpdateTemplateKHR = (PFN_vkCreateDescriptorUpdateTemplateKHR)vkGetDeviceProcAddr(vkdev->GetDevice(), "vkCreateDescriptorUpdateTemplateKHR");

		vkdev->CreateDescriptorUpdateTemplate(&descriptor_update_template_create_info, &descriptor_update_template);
		//VkResult ret = vkCreateDescriptorUpdateTemplateKHR(vkdev->GetDevice(), &descriptor_update_template_create_info, 0, &descriptor_update_template);
		//CHECK_EQ(VK_SUCCESS, ret) << "vkCreateDescriptorUpdateTemplateKHR failed " << ret;
	}
}