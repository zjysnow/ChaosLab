#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	// A*B
	class CHAOS_API GEMM : public Layer
	{
	public:
		GEMM();

		virtual void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt = Option()) const override;
	};
}