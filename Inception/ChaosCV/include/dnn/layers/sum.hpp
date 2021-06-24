#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	inline namespace dnn
	{
		class CHAOS_API Sum : public Layer
		{
		public:
			Sum();
			void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const override;

			void Set(const std::string& pname, const std::any& param) override;

			bool all = false;
			Array<uint32> vecdim = { 0 };
		};
	}
}