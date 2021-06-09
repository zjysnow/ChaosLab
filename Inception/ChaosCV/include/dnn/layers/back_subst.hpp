#pragma once

#include "dnn/layer.hpp"

namespace chaos::inline dnn
{
	// here just from SVD::backsubst
	class CHAOS_API BackSubst : public Layer
	{
	public:
		BackSubst();
		void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const override;
	};
}