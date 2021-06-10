#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	inline namespace dnn
	{
		class CHAOS_API Invert : public Layer
		{
		public:
			enum Method
			{
				DECOMP_LU,
				DECOMP_SVD,
			};

			Invert();
			void CreatePipeline(const Option&) override;
			void DestroyPipeline(const Option&) override;

			void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const override;

			void Set(const std::string& pname, const std::any& param) override;

			Ptr<Layer> svd;
			Ptr<Layer> backsubst;

			int method = DECOMP_SVD;
		};
	}
}