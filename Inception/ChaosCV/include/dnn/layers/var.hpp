#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	inline namespace dnn
	{
		class CHAOS_API Var : public Layer
		{
		public:
			Var();

			void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const override;

			void Set(const std::string& pname, const std::any& param) override;

			bool all;
			Array<int> vecdim = { 0 };
			bool unbias = true;

			Ptr<Layer> mean;
			Ptr<Layer> sub2;
		};
	}
}