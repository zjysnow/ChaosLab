#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	// p-Norm
	class CHAOS_API Norm : public Layer
	{
	public:
		Norm();

		void Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const override;

		void Set(const std::string& pname, const std::any& val) override;
		
		float p = 2.f;
	};
}