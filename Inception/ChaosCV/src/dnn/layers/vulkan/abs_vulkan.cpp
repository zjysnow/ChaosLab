#include "dnn/layers/vulkan/abs_vulkan.hpp"
#include "dnn/layers/vulkan/shaders/abs_spv_data.hex.hpp"

namespace chaos
{
	AbsVulkan::AbsVulkan() : Abs()
	{
		support_vulkan = true;
	}

	void AbsVulkan::CreatePipeline(const Option&)
	{
		std::vector<VulkanSpecializationType> specializations;
		pipeline_abs = new ComputePipeline(vkdev);
		pipeline_abs->SetOptimalLocalSize(std::min(64,1),1,1);
		pipeline_abs->Create(abs_spv_data, sizeof(abs_spv_data), "abs", specializations, 1, 3);
	}

	void AbsVulkan::DestroyPipeline(const Option&)
	{
		delete pipeline_abs;
		pipeline_abs = nullptr;
	}

	void AbsVulkan::Forward(std::vector<VulkanTensor>& bottom_top_blobs, ComputeCommand& cmd, const Option&) const
	{
		uint32 total = (uint32)bottom_top_blobs[0].total();
		std::vector<VulkanConstantType> constants(3);
		constants[0].i = 1; // bottom_top_blobs[0].shape[0];
		constants[1].i = 1; // bottom_top_blobs[0].shape[1];
		constants[2].i = total; // bottom_top_blobs[0].shape[2];

		Shape dispatcher(1,1,total);

		cmd.RecordPipeline(pipeline_abs, bottom_top_blobs, constants, dispatcher);
	}
}