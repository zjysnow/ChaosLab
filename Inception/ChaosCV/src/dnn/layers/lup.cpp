#include "dnn/layers/decomp.hpp"

namespace chaos
{
	LUP::LUP() : Decomp("LUP") {}

	void LUP::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const
	{
		CHECK_EQ(1, bottom_blobs.size()) << "layer '" << type << "' expect 1 input but got " << bottom_blobs.size();
		const Tensor& A = bottom_blobs[0];

		CHECK_EQ(2, A.shape.dims) << "input A should be a matrix";
		CHECK_EQ(A.shape[0], A.shape[1]) << "input A should be a square matrix";
		CHECK_EQ(3, top_blobs.size()) << "layer '" << type << "' expect 3 outputs but got " << top_blobs.size();

		Tensor& L = top_blobs[0];
		Tensor& U = top_blobs[1];
		Tensor& P = top_blobs[2];

		if (L.empty()) L.Create(A.shape, A.shape.steps(), DataType::D4, Packing::CHW, opt.blob_allocator);
		if (U.empty()) U.Create(A.shape, A.shape.steps(), DataType::D4, Packing::CHW, opt.blob_allocator);
		if (P.empty()) P.Create(A.shape, A.shape.steps(), DataType::D4, Packing::CHW, opt.blob_allocator);

		uint32 n = A.shape[0];

		uint32 ustep = U.steps[0];
		uint32 pstep = P.steps[0];
		uint32 lstep = L.steps[0];

		// to preset U, L, P
		A.CopyTo(U);
		memset(L.data, 0, sizeof(float) * L.total());
		memset(P.data, 0, sizeof(float) * P.total());
		for (size_t i = 0; i < n; i++)
		{
			L[i * lstep + i] = P[i * pstep + i] = 1.f;
		}

		
		size_t k;
		for (size_t i = 0; i < n - 1; i++)
		{
			k = i;
			// select k(>=j) that maximizes |u_{ij}|
			for (size_t j = i + 1; j < n; j++)
			{
				if (std::abs(U[j * ustep + i]) > std::abs(U[k * ustep + i])) k = j;
			}

			CHECK_GT(std::abs(U[k * ustep + i]), 1e-5) << "major is zero";

			if (k != i)
			{
				// interchange rows of U: u(i,i:n)<-->u(k,i:n)
				for (size_t j = i; j < n; j++)
				{
					std::swap(U[i * ustep + j], U[k * ustep + j]);
				}
				// interchange rows of P: p(i,:)<-->p(k,:)
				for (size_t j = 0; j < n; j++)
				{
					std::swap(P[i * pstep + j], P[k * pstep + j]);
				}
				// interchange rows of L: l(i,0:i-1)<-->l(k,0:i-1)
				for (size_t j = 0; j < i; j++)
				{
					std::swap(L[i * lstep + j], L[k * lstep + j]);
				}
			}

			float d = 1 / U[i * ustep + i];
			for (size_t j = i + 1; j < n; j++)
			{
				float a = U[j * ustep + i] * d;
				L[j * lstep + i] = a; // L_{ji} = U_{ji} / U_{ii}
				for (size_t k = i; k < n; k++)
				{
					U[j * ustep + k] -= a * U[i * ustep + k];
				}
			}
		}
	}
}