#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	inline namespace dnn
	{
		class CHAOS_API Abs : public Layer
		{
		public:
			Abs();

			void Forward(std::vector<Tensor>& bottom_top_blobs, const Option& opt) const override;
		};
	}
}