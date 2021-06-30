#pragma once

#include "core/core.hpp"
#include "core/tensor.hpp"

#include "dnn/layer.hpp"
#include "dnn/layer_factory.hpp"

namespace chaos
{
	class CHAOS_API PCA
	{
	public:
		PCA();

		Tensor mean;
		Tensor u;

		Ptr<Layer> svd;
		Ptr<Layer> gemm;
	};
}