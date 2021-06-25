#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	inline namespace dnn
	{
		class CHAOS_API Permute : public Layer
		{
		public:
			Permute();

			void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const override;

			void Set(const std::string& pname, const std::any& val) override;

			Array<int> orders;
		};
	}
}