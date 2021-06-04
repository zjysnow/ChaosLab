#pragma once

#include "dnn/layers/abs.hpp"

namespace chaos
{
	class CHAOS_API AbsVulkan : public Abs
	{
	public:
		AbsVulkan();

		void CreatePipeline() override;
		void DestroyPipeline() override;

		void Forward(std::vector<VulkanTensor>& bottom_top_blobs, ComputeCommand& cmd) const;

		ComputePipeline* pipeline_abs;
	};
}