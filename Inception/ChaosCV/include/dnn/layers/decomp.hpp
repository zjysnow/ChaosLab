#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	class CHAOS_API Decomp : public Layer
	{
	public:
		enum
		{
			LUP,
			CHOLESKY,
			SVD,
			EIG,
			QR,
		};

		using Layer::Layer;
	};

	// P[A|y] = [LU|y']
	class CHAOS_API LUP : public Decomp
	{
	public:
		LUP();

		virtual void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt = Option()) const override;
	};

	class Cholesky : public Decomp
	{
	public:

	};

	class CHAOS_API SVD : public Decomp
	{
	public:
		enum
		{
			/** allow the algorithm to modify the decomposed matrix; it can save space and speed up
			processing. currently ignored. */
			MODIFY_A = 1,
			/** indicates that only a vector of singular values `w` is to be processed, while u and vt
			will be set to empty matrices */
			NO_UV = 2,
			/** when the matrix is not square, by default the algorithm produces u and vt matrices of
			sufficiently large size for the further A reconstruction; if, however, FULL_UV flag is
			specified, u and vt will be full-size square orthogonal matrices.*/
			FULL_UV = 4
		};

		SVD();

		virtual void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const override;

		int flags = MODIFY_A;
	};

	class CHAOS_API Eigen : public Decomp
	{
	public:
		Eigen();

		virtual void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const override;
	};
}