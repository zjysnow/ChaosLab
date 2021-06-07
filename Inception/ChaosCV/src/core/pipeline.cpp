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

	void Pipeline::CreateDescriptorSetLayout(const size_t& binding_count, const ShaderStageFlag* flags, const VkDescriptorType* types)
	{
		std::vector<VkDescriptorSetLayoutBinding> bingdings(binding_count);
		for (uint32 i = 0; i < binding_count; i++)
		{
			bingdings[i].binding = i;
			bingdings[i].descriptorCount = 1;
			bingdings[i].descriptorType = types[i]; //VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
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







	static void InjectLocalSize(const uint32* code, size_t size, uint32 local_size_x, uint32 local_size_y, uint32 local_size_z, uint32* new_code, size_t* new_size)
	{
		uint32 local_size_x_id = -1;
		uint32 local_size_y_id = -1;
		uint32 local_size_z_id = -1;
		uint32 gl_WorkGroupSize_id = -1;

		const uint32* p = code;
		uint32* dp = new_code;

		// skip magic version generator bound schema
		memcpy(dp, p, 5 * sizeof(uint32));
		p += 5;
		dp += 5;

		// foreach op
		while ((const uchar*)p < (const uchar*)code + size)
		{
			uint32 op_code = p[0];

			uint16 word_count = op_code >> 16; // p[1] high
			uint16 op = op_code & 0xFFFF; // p[1] low

			if (op == 16) // OpExecutionMode
			{
				uint32 mode = p[2];
				if (mode == 17) // LocalSize
				{
					memcpy(dp, p, word_count * sizeof(uint32));

					// set local_size_xyz
					dp[3] = local_size_x;
					dp[4] = local_size_y;
					dp[5] = local_size_z;

					p += word_count;
					dp += word_count;
					continue;
				}
			}
			else if (op == 50) // OpSpecConstant
			{
				uint32 id = p[2];
				if (id == local_size_x_id || id == local_size_y_id || id == local_size_z_id)
				{
					p += word_count;
					continue;
				}
			}
			else if (op == 51) // OpSpecConstantComposite
			{
				uint32 id = p[2];
				if (id == gl_WorkGroupSize_id)
				{
					if (word_count == 6 && (p[3] == local_size_x_id || p[4] == local_size_y_id || p[5] == local_size_z_id))
					{
						p += word_count;
						continue;
					}
				}
			}
			else if (op == 71) // OpDecorate
			{
				uint32 id = p[1];
				uint32 decoration = p[2];
				// removed decoration == 1
				if (decoration == 11) // BuiltIn
				{
					uint32 builtin = p[3];
					if (builtin == 25) // WorkgroupSize
					{
						gl_WorkGroupSize_id = id;
						p += word_count;
						continue;
					}
				}
			}

			memcpy(dp, p, word_count * sizeof(uint32));
			p += word_count;
			dp += word_count;
		}
		*new_size = (uchar*)dp - (uchar*)new_code;
	}

	ComputePipeline::ComputePipeline(const VulkanDevice* vkdev) : Pipeline(vkdev) {}
	ComputePipeline::~ComputePipeline()
	{
		if (descriptor_update_template) vkdev->DestroyDescriptorUpdateTemplate(descriptor_update_template);
		vkDestroyShaderModule(vkdev->GetDevice(), comp, nullptr);
	}

	void ComputePipeline::Create(const uint32* comp_data, size_t comp_size, const char* entry_name, const std::vector<VulkanSpecializationType>& specializations, size_t binding_count, size_t push_constant_count)
	{
		uint32* comp_data_modefied = new uint32[comp_size]();
		size_t comp_size_modefied = comp_size;
		InjectLocalSize(comp_data, comp_size, local_size_x, local_size_y, local_size_z, comp_data_modefied, &comp_size_modefied);
		comp = CompileShaderModule(comp_data_modefied, comp_size_modefied);

		delete[] comp_data_modefied;

		std::vector<VkDescriptorType> descriptor_types(binding_count, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
		std::vector<ShaderStageFlag> shader_stage_flags(binding_count, VK_SHADER_STAGE_COMPUTE_BIT);
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

	void ComputePipeline::SetOptimalLocalSize(uint32 x, uint32 y, uint32 z)
	{
		if (x == 0 and y == 0 and z == 0)
		{
			x = y = z = 4;
		}

		x = std::min(x, vkdev->info.max_workgroup_count_x);
		y = std::min(y, vkdev->info.max_workgroup_count_y);
		z = std::min(z, vkdev->info.max_workgroup_count_z);

		if (x * y * z > vkdev->info.max_workgroup_invocations) // why ?
		{
			uint32 max_xy = std::max(1u, (uint32)sqrt(vkdev->info.max_workgroup_invocations / z));
			while (x * y >= max_xy)
			{
				x = std::max(1u, x / 2);
				y = std::max(1u, y / 2);
			}
		}

		local_size_x = x;
		local_size_y = y;
		local_size_z = z;
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
		descriptor_update_template_create_info.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR; 
		descriptor_update_template_create_info.descriptorSetLayout = descriptorset_layout;
		descriptor_update_template_create_info.pipelineBindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
		descriptor_update_template_create_info.pipelineLayout = pipeline_layout;

		vkdev->CreateDescriptorUpdateTemplate(&descriptor_update_template_create_info, &descriptor_update_template);
	}
}