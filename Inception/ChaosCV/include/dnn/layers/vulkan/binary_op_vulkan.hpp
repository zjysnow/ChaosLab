#pragma once

#include "dnn/layers/binary_op.hpp"

namespace chaos
{
	inline namespace dnn
	{
		class BinaryOpVulkan : public BinaryOp
		{
		public:
			BinaryOpVulkan();

			void CreatePipeline(const Option& opt) override;
			void DestroyPipeline(const Option& opt) override;

			void Forward(const std::vector<VulkanTensor>& bottom_blobs, std::vector<VulkanTensor>& top_blobs, ComputeCommand& cmd, const Option& opt) const override;

			ComputePipeline* binary_op_pipeline;
		};
	}
}