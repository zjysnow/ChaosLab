#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	class Sum : public Layer
	{
	public:
		Sum();

		void Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const override;
		void Set(const std::string& pname, const std::any& val);

		int dim = 0;
	};
}