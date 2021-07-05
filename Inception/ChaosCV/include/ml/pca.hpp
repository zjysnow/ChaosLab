#pragma once

#include "core/core.hpp"
#include "core/tensor.hpp"

namespace chaos
{
	class CHAOS_API PCA
	{
	public:
		enum
		{
			ROW_MAJOR,
			COL_MAJOR,
		};

		PCA();

		PCA& operator()(const Tensor& data, int flags);

		Tensor Project(const Tensor& data);
		Tensor BackProject(const Tensor data);

	private:
		Tensor M;
		Tensor P;
	};
}