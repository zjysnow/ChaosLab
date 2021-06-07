#pragma once

#include "dnn/layers/abs.hpp"

namespace chaos
{
	class CHAOS_API AbsVulkan : virtual public Abs
	{
	public:
		AbsVulkan();

		void CreatePipeline(const Option& opt) override;
		void DestroyPipeline(const Option& opt) override;

		void Forward(std::vector<VulkanTensor>& bottom_top_blobs, ComputeCommand& cmd, const Option& opt) const;

		ComputePipeline* pipeline_abs;
	};
}