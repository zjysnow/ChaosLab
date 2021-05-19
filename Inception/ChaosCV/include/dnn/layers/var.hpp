#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	class CHAOS_API Var : public Layer
	{
	public:
		Var();

		void Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const override;
		void Set(const std::string& pname, const std::any& val) override;

		bool all = false;
		bool unbiased = true;
		std::vector<uint32> vecdim = { 0 };
	};
}