#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	// Y = alpha*A*B+beta*C
	class CHAOS_API GEMM : public Layer
	{
	public:
		GEMM();

		virtual void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt = Option()) const override;

		void Set(const std::string& name, const std::any& val);


		float alpha = 1.f;
		float beta = 0.f;
		//bool transA = false;
		//bool transB = false;
	};
}