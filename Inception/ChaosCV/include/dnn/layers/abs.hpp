#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	class CHAOS_API Abs : public Layer
	{
	public:
		Abs();

		void Forward(std::vector<Tensor>& bottom_top_blobs) const override;
		//void Forward(std::vector<VulkanTensor>& bottom_top_blobs, ComputeCommand& cmd) const override;
	};
}