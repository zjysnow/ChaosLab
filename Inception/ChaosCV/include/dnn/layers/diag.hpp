#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	class Diag : public Layer
	{
	public:
		Diag();

		virtual void Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const override;

		int k = 0; // not use now
	};
}