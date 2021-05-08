#pragma once

#include "dnn/layer.hpp"
#include "dnn/layers/decomp.hpp"

namespace chaos
{
	class Backsubst : public Layer
	{
	public:
		Backsubst();

		void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const override;

		// see Decmop enum
		int flag = Decomp::SVD;
	};
}