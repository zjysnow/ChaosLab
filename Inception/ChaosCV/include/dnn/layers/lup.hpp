#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	namespace dnn
	{
		class CHAOS_API LUP : public Layer
		{
		public:
			LUP();

			void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const override;
		};
	}
}