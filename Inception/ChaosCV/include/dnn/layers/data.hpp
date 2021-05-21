#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	namespace dnn
	{
		class CHAOS_API Data : public Layer
		{
		public:
			Data();
			void Forward(std::vector<Tensor>& bottom_top_blobs, const Option& opt) const override;

			//size_t bottoms_count() const final { return 1; }
			//size_t tops_count() const final { return 1; }
		};
	}
}