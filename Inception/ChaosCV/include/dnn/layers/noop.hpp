#pragma once

#include "dnn/layer.hpp"

namespace chaos::inline dnn
{
	class CHAOS_API Noop : public Layer
	{
	public:
		Noop();

		void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const override;
		void Forward(std::vector<Tensor>& bottom_top_blobs, const Option& opt) const override;

		void Forward(const std::vector<VulkanTensor>& bottom_blobs, std::vector<VulkanTensor>& top_blobs, ComputeCommand& cmd, const Option& opt) const override;
		void Forward(std::vector<VulkanTensor>& bottom_top_blobs, ComputeCommand& cmd, const Option& opt) const override;
	};
}