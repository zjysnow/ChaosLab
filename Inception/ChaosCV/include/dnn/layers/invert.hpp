#pragma once

#include "dnn/layer.hpp"
#include "dnn/layers/decomp.hpp"

namespace chaos
{
	class Invert : public Layer
	{
	public:
		Invert();

		void Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const override;

		// see Decmop enum
		int method = Decomp::SVD;

		float eps = FLT_EPSILON; //1e-5f;

		Ptr<Decomp> decomp;
	};
}