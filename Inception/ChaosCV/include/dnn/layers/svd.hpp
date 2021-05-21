#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	namespace dnn
	{
		class CHAOS_API SVD : public Layer
		{
		public:
			enum UV
			{
				/** indicates that only a vector of singular values `w` is to be processed, while u and vt
				will be set to empty matrices */
				NO_UV = 2,
				/** when the matrix is not square, by default the algorithm produces u and vt matrices of
				sufficiently large size for the further A reconstruction; if, however, FULL_UV flag is
				specified, u and vt will be full-size square orthogonal matrices.*/
				FULL_UV = 4
			};

			SVD();

			void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const override;
			void Set(const std::string& pname, const std::any& val) final;

			int uv = 1;
		};
	}
}