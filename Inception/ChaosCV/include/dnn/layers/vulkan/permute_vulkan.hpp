#pragma once

#include "dnn/layers/permute.hpp"

namespace chaos
{
	inline namespace dnn
	{
		class CHAOS_API PermuteVulkan : public Permute
		{
		public:
			PermuteVulkan();

			void CreatePipeline(const Option& opt) override;
			void DestroyPipeline(const Option& opt) override;

			void Forward(const std::vector<VulkanTensor>& bottom_blobs, std::vector<VulkanTensor>& top_blobs, ComputeCommand& cmd, const Option& opt) const;

			ComputePipeline* permute_pipeline;
		};
	}
}