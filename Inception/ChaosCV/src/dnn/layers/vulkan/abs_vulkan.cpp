#include "dnn/layers/vulkan/abs_vulkan.hpp"
#include "dnn/layers/vulkan/shaders/abs_spv_data.hex.hpp"

namespace chaos
{
	inline namespace dnn
	{
		AbsVulkan::AbsVulkan() : Abs()
		{
			support_vulkan = true;
		}

		void AbsVulkan::CreatePipeline(const Option&)
		{
			std::vector<VulkanSpecializationType> specializations;
			abs_pipeline = new ComputePipeline(vkdev);
			abs_pipeline->SetOptimalLocalSize(std::min(64, 1), 1, 1);
			abs_pipeline->Create(abs_spv_data, sizeof(abs_spv_data), "abs", specializations, 1, 1);
		}

		void AbsVulkan::DestroyPipeline(const Option&)
		{
			delete abs_pipeline;
			abs_pipeline = nullptr;
		}

		void AbsVulkan::Forward(std::vector<VulkanTensor>& bottom_top_blobs, ComputeCommand& cmd, const Option&) const
		{
			uint32 total = (uint32)bottom_top_blobs[0].total();
			std::vector<VulkanConstantType> constants(1);
			constants[0].i = total; // bottom_top_blobs[0].shape[0];

			Shape dispatcher(1, 1, total);

			cmd.RecordPipeline(abs_pipeline, bottom_top_blobs, constants, dispatcher);
		}
	}
}