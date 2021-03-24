#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	class Noop : public Layer
	{
	public:
		Noop();

		virtual void Forward(std::vector<Tensor>& bottom_top_blobs, const Option& opt) const override;
	};
}