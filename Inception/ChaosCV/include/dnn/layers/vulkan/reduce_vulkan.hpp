#pragma once

#include "dnn/layers/reduce.hpp"

namespace chaos
{
	inline namespace dnn
	{
		class ReduceVulkan : virtual public Reduce
		{
		public:
			ReduceVulkan();

			void CreatePipeline(const Option& opt) override;
			void DestroyPipeline(const Option& opt) override;

			void Forward(const std::vector<VulkanTensor>& bottom_blobs, std::vector<VulkanTensor>& top_blobs, ComputeCommand& cmd, const Option& opt) const override;

			ComputePipeline* reduce_pipeline;
		};
	}
}