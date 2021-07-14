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

	void Pipeline::CreateDescriptorSetLayout(const size_t& binding_count, const VkShaderStageFlagBits* flags, const VkDescriptorType* types)
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
		std::vector<VkShaderStageFlagBits> shader_stage_flags(binding_count, VK_SHADER_STAGE_COMPUTE_BIT);
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








	GraphicsPipeline::GraphicsPipeline(const VulkanDevice* vkdev) : Pipeline(vkdev)
	{
	}
	GraphicsPipeline::~GraphicsPipeline()
	{
		vkDestroyShaderModule(vkdev->GetDevice(), vert, nullptr);
		vkDestroyShaderModule(vkdev->GetDevice(), frag, nullptr);

		vkDestroyRenderPass(vkdev->GetDevice(), render_pass, nullptr);
	}

	void GraphicsPipeline::CreateRenderPass(VkFormat format)
	{
		VkAttachmentDescription color_attachment{};
		color_attachment.format = format;
		color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference color_attachment_ref{};
		color_attachment_ref.attachment = 0;
		color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &color_attachment_ref;

		VkRenderPassCreateInfo render_pass_create_info{};
		render_pass_create_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		render_pass_create_info.attachmentCount = 1;
		render_pass_create_info.pAttachments = &color_attachment;
		render_pass_create_info.subpassCount = 1;
		render_pass_create_info.pSubpasses = &subpass;

		VkResult ret = vkCreateRenderPass(vkdev->GetDevice(), &render_pass_create_info, nullptr, &render_pass);
		CHECK_EQ(VK_SUCCESS, ret) << "vkCreateRenderPass failed " << ret;
	}

	void GraphicsPipeline::Create(const uint32* vert_data, size_t vert_size, const uint32* frag_data, size_t frag_size,
		VkExtent2D extent, VkFormat format, VkPolygonMode polygon_mode, VkFrontFace front_face, VkPrimitiveTopology topoloty, VkCullModeFlagBits cull_mode)
	{
		// create shader module 
		vert = CompileShaderModule(vert_data, vert_size);
		frag = CompileShaderModule(frag_data, frag_size);

		// create descriptorset layout
		VkDescriptorType descriptor_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		VkShaderStageFlagBits shader_stage_flag = VK_SHADER_STAGE_VERTEX_BIT;
		CreateDescriptorSetLayout(1, &shader_stage_flag, &descriptor_type);

		// create render pass
		CreateRenderPass(format);

		// create pipeline layout
		CreatePipelineLayout();

		// create pipeline
		VkPipelineShaderStageCreateInfo vert_shader_stage_info{};
		vert_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vert_shader_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vert_shader_stage_info.module = vert;
		vert_shader_stage_info.pName = "main";

		VkPipelineShaderStageCreateInfo frag_shader_stage_info{};
		frag_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		frag_shader_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		frag_shader_stage_info.module = frag;
		frag_shader_stage_info.pName = "main";

		VkPipelineShaderStageCreateInfo shader_stages[2]{ vert_shader_stage_info, frag_shader_stage_info };

		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(float) * 5;
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[0].offset = 0; //offsetof(Vertex, pos);
		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = 2 * sizeof(float); // offsetof(Vertex, color);


		VkPipelineVertexInputStateCreateInfo vertex_input_info{};
		vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertex_input_info.pVertexBindingDescriptions = &bindingDescription;
		vertex_input_info.vertexBindingDescriptionCount = 1;

		vertex_input_info.pVertexAttributeDescriptions = attributeDescriptions.data();
		vertex_input_info.vertexAttributeDescriptionCount = 2;

		VkPipelineInputAssemblyStateCreateInfo input_assembly{};
		input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		input_assembly.topology = topoloty; // VK_PRIMITIVE_TOPOLOGY_LINE_LIST; // VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; //VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		input_assembly.primitiveRestartEnable = VK_FALSE;

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)extent.width;
		viewport.height = (float)extent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = extent;

		VkPipelineViewportStateCreateInfo viewport_state{};
		viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewport_state.viewportCount = 1;
		viewport_state.pViewports = &viewport;
		viewport_state.scissorCount = 1;
		viewport_state.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = polygon_mode; // VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = cull_mode; // VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = front_face; //VK_FRONT_FACE_CLOCKWISE; // VK_FRONT_FACE_COUNTER_CLOCKWISE; //VK_FRONT_FACE_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;

		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState color_blend_attachment{};
		color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		color_blend_attachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo color_blending{};
		color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		color_blending.logicOpEnable = VK_FALSE;
		color_blending.logicOp = VK_LOGIC_OP_COPY;
		color_blending.attachmentCount = 1;
		color_blending.pAttachments = &color_blend_attachment;
		color_blending.blendConstants[0] = 0.0f;
		color_blending.blendConstants[1] = 0.0f;
		color_blending.blendConstants[2] = 0.0f;
		color_blending.blendConstants[3] = 0.0f;


		VkGraphicsPipelineCreateInfo pipeline_info{};
		pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeline_info.stageCount = 2;
		pipeline_info.pStages = shader_stages;
		pipeline_info.pVertexInputState = &vertex_input_info;
		pipeline_info.pInputAssemblyState = &input_assembly;
		pipeline_info.pViewportState = &viewport_state;
		pipeline_info.pRasterizationState = &rasterizer;
		pipeline_info.pMultisampleState = &multisampling;
		pipeline_info.pColorBlendState = &color_blending;
		pipeline_info.layout = pipeline_layout;
		pipeline_info.renderPass = render_pass;
		pipeline_info.subpass = 0;
		pipeline_info.basePipelineHandle = VK_NULL_HANDLE;

		VkResult ret = vkCreateGraphicsPipelines(vkdev->GetDevice(), VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline);
		CHECK_EQ(VK_SUCCESS, ret) << "vkCreateGraphicsPipelines failed " << ret;
	}
}