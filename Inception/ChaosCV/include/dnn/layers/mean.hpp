#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	class Mean : public Layer
	{
	public:
		Mean();

		void Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const override;

		void Set(const std::string& pname, const std::any& val) override;
		// -1 means all
		int dim = 0;
	};
}