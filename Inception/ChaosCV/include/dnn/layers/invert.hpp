#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	inline namespace dnn
	{
		class SVD;
		class Backsubst;
		class CHAOS_API Invert : public Layer
		{
		public:
			enum Method
			{
				DECOMP_SVD,
				DECOMP_LU,
			};
			Invert();

			void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const override;
			void Set(const std::string& pname, const std::any& param) override;

			int method = DECOMP_SVD;
			Ptr<SVD> svd;
			Ptr<Backsubst> backsubst;
		};
	}
}