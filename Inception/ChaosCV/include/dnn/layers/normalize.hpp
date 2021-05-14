#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	// treat the tensor as a vector but do not reshape
	class Normalize : public Layer
	{
	public:
		enum Method
		{
			ZSCORE, // center and scale to have mean 0 and standard deviation 1
			NORM,	// p-norm or infinity norm
			RANGE,	// rescale range of data to an interval of the form [f1, f2] (default is [0, 1]), where f1 < f2
		};

		Normalize();

		void Forward(const Tensor& bottom_blob,Tensor& top_blob, const Option& opt) const override;
		void Set(const std::string& pname, const std::any& val) override;

		int method = ZSCORE;
		/// <summary>
		/// <para>1. method == NORM, f1 used for p of p-norm</para>
		/// <para>2. method == RANGE, f1 used for min of range</para>
		/// </summary>
		float f1 = 0;
		float f2 = 0;
		int dim = 0;
	};
}