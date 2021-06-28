#pragma once

#include "dnn/layers/permute.hpp"

namespace chaos
{
	inline namespace dnn
	{
		class PermuteVulkan : virtual public Permute
		{
		public:
			PermuteVulkan();

			void CreatePipeline(const Option& opt) override;
			void DestroyPipeline(const Option& opt) override;

			void Forward(const std::vector<VulkanTensor>& bottom_blobs, std::vector<VulkanTensor>& top_blobs, ComputeCommand& cmd, const Option& opt) const override;

			ComputePipeline* permute_pipeline;
			ComputePipeline* transpose_pipeline;
		};
	}
}