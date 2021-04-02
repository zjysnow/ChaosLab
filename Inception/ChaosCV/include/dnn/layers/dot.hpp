#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	class Dot : public Layer
	{
	public:
		Dot();

		virtual void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const override;
	};
}