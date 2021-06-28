#pragma once

#include "dnn/layers/gemm.hpp"

namespace chaos
{
	inline namespace dnn
	{
		class GEMMVulkan : virtual public GEMM 
		{
		public:
			GEMMVulkan();

			void CreatePipeline(const Option& opt) override;
			void DestroyPipeline(const Option& opt) override;

			void Forward(const std::vector<VulkanTensor>& bottom_blobs, std::vector<VulkanTensor>& top_blobs, ComputeCommand& cmd, const Option& opt) const override;

			ComputePipeline* gemm_pipeline;
		};
	}
}