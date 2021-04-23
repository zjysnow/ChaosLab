#include "core/pipeline.hpp"

namespace chaos
{
	Pipeline::Pipeline(const VulkanDevice* vkdev) : vkdev(vkdev) {}
	Pipeline::~Pipeline()
	{
		//vkDestroyDescriptorPool(vkdev->GetDevice(), descriptor_pool, nullptr);
		vkDestroyDescriptorSetLayout(vkdev->GetDevice(), descriptor_set_layout, nullptr);
		vkDestroyPipeline(vkdev->GetDevice(), pipeline, nullptr);
		vkDestroyPipelineLayout(vkdev->GetDevice(), pipeline_layout, nullptr);
	}


	void Pipeline::CreateDescriptorSetLayout(const uint32& binding_count, const VkShaderStageFlagBits* flags, const VkDescriptorType* types)
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
		
		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = binding_count;
		layoutInfo.pBindings = bingdings.data();

		auto ret = vkCreateDescriptorSetLayout(vkdev->GetDevice(), &layoutInfo, nullptr, &descriptor_set_layout);
		CHECK_EQ(VK_SUCCESS, ret) << "vkCreateDescriptorSetLayout failed " << ret;
	}


	void Pipeline::CreatePipelineLayout(int push_constant_count)
	{
		VkPipelineLayoutCreateInfo pipeline_layout_info{};
		pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		if (descriptor_set_layout)
		{
			pipeline_layout_info.setLayoutCount = 1;
			pipeline_layout_info.pSetLayouts = &descriptor_set_layout;
		}

		if (push_constant_count > 0)
		{
			VkPushConstantRange push_constant_range{};
			push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			push_constant_range.size = sizeof(VkConstantType) * push_constant_count;

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

	GraphicsPipeline::GraphicsPipeline(const VulkanDevice* vkdev) : Pipeline(vkdev)
	{
		descriptor_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		shader_stage_flag = VK_SHADER_STAGE_VERTEX_BIT;
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
		color_attachment.format = (VkFormat)format;
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
		VkFormat format, uint32 width, uint32 height, VkPolygonMode polygon_mode, VkFrontFace front_face, VkPrimitiveTopology topoloty)
	{
		// create shader module 
		vert = CompileShaderModule(vert_data, vert_size);
		frag = CompileShaderModule(frag_data, frag_size);

		// create descriptorset layout
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
		viewport.width = (float)width;
		viewport.height = (float)height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent.width = width; // extent;
		scissor.extent.height = height;

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
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
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