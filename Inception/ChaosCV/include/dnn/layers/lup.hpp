#pragma once

#include "dnn/layers/decomp.hpp"

namespace chaos
{
	// P[A|y] = [LU|y']
	class LUP : public Decomp
	{
	public:
		LUP();

		virtual void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt = Option()) const override;
	};
}