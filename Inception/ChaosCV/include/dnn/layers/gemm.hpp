#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	namespace dnn
	{
		// General Matrix Multiplication
		// C = alpha * A * B + beta * C
		class CHAOS_API GEMM : public Layer
		{
		public:
			GEMM();

			void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const override;

			void Set(const std::string& pname, const std::any& val) final;

			float alpha = 1.f;
			float beta = 0.f;

			bool transA = false;
			bool transB = false;
			bool transC = false;
		};
	}
}