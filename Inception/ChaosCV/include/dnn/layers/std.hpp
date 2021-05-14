#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	class STD : public Layer
	{
	public:
		STD();

		void Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const override;

		int dim = 0;
	};
}