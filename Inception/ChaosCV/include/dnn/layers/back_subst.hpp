#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	inline namespace dnn
	{
		// here just from SVD::backsubst
		class CHAOS_API BackSubst : public Layer
		{
		public:
			BackSubst();
			void CreatePipeline(const Option& opt) override;
			void DestroyPipeline(const Option& opt) override;

			void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const override;

			Ptr<Layer> gemm;
			Ptr<Layer> permute;
		};
	}
}