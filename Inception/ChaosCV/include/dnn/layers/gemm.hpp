#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	inline namespace dnn
	{
		class GEMM : public Layer
		{
		public:
			GEMM();

			void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const override;

			void Set(const std::string& pname, const std::any& param) override;

			bool transA = false;
			bool transB = false;

			float alpha = 1.f;
			float beta = 0.f;
		};
	}
}