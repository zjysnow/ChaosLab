#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	inline namespace dnn
	{
		class CHAOS_API Mean : public Layer
		{
		public:
			Mean();

			void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const override;

			void Set(const std::string& pname, const std::any& param) override;

			bool all = false;
			Array<int> vecdim = { 0 };

			Ptr<Layer> sum;
			Ptr<Layer> div;
		};
	}
}