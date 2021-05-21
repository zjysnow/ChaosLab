#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	namespace dnn
	{
		class CHAOS_API Noop : public Layer
		{
		public:
			Noop();
			void Forward(std::vector<Tensor>& bottom_top_blobs, const Option& opt) const override;
		};
	}
}