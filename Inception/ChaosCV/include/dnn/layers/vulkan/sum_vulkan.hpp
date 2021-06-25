#pragma once

#include "dnn/layers/sum.hpp"

namespace chaos
{
	inline namespace dnn
	{
		class SumVulkan : public Sum
		{
		public:
			SumVulkan();

			void CreatePipeline(const Option& opt) override;
			void DestroyPipeline(const Option& opt) override;

			void Forward(const std::vector<VulkanTensor>& bottom_blobs, std::vector<VulkanTensor>& top_blobs, ComputeCommand& cmd, const Option& opt) const override;

			ComputePipeline* sum_pipeline;
		};
	}
}