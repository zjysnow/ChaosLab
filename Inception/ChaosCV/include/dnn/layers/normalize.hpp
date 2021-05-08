#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	// treat the tensor as a vector but do not reshape
	class Normalize : public Layer
	{
	public:
		Normalize();

		enum NormType
		{
			L1,
			L2,
			MINMAX,
		};

		void Forward(const Tensor& bottom_blob,Tensor& top_blob, const Option& opt) const override;
		void Set(const std::string& pname, const std::any& val) override;

		int norm_type = L2;
	};
}