#include "dnn/layers/vulkan/abs_vulkan.hpp"
#include "dnn/layers/vulkan/shaders/abs_spv_data.hex.hpp"

namespace chaos
{
	AbsVulkan::AbsVulkan() : Abs() 
	{
		support_vulkan = true;
	}

	void AbsVulkan::CreatePipeline()
	{
		std::vector<VulkanSpecializationType> specializations;
		pipeline_abs = new ComputePipeline(vkdev);
		pipeline_abs->Create(abs_spv_data, sizeof(abs_spv_data), "abs", specializations, 1, 3);
	}

	void AbsVulkan::DestroyPipeline()
	{
		delete pipeline_abs;
	}

	void AbsVulkan::Forward(std::vector<VulkanTensor>& bottom_top_blobs, ComputeCommand& cmd) const
	{
		std::vector<VulkanConstantType> constants(3);
		constants[0].i = bottom_top_blobs[0].shape[0];
		constants[1].i = bottom_top_blobs[0].shape[1];
		constants[2].i = bottom_top_blobs[0].shape[2];

		cmd.RecordPipeline(pipeline_abs, bottom_top_blobs, constants, bottom_top_blobs[0].shape);
	}
}