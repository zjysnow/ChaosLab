#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	// alpha*A*B+beta*C
	class CHAOS_API GEMM : public Layer
	{
	public:
		GEMM();

		virtual void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt = Option()) const override;

		float alpha = 1.f;
		float beta = 0.f;
	};
}