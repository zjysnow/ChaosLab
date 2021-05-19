#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	class Permute : public Layer
	{
	public:
		Permute();

		void Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const override;

		void Set(const std::string& name, const std::any& val) override;

		std::vector<uint32> orders;
	};
}