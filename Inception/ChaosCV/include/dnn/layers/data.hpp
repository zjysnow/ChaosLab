#pragma once
#include "dnn/layer.hpp"

namespace chaos
{
	class Data : public Layer
	{
	public:
		Data();

		void Forward(Tensor& bottom_top_blob, const Option& opt) const override;

		size_t top_blobs_num() const final { return 1; }
	};
}