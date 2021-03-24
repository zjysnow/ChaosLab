#include "dnn/layers/lup.hpp"

namespace chaos
{
	void LUImpl(const Tensor& A, Tensor& P, Tensor& L, Tensor& U, float eps = 1e-5)
	{
		uint32 n = A.shape[0];

		// to preset U, L, P
		A.CopyTo(U);
		memset(L.data, 0, n * n * sizeof(float));
		memset(P.data, 0, n * n * sizeof(float));
		for (uint32 i = 0; i < n; i++)
		{
			L[i * n + i] = P[i * n + i] = 1.f;
		}

		size_t rstep = U.steps[0];
		uint32 k;
		for (uint32 i = 0; i < n - 1; i++)
		{
			k = i;
			// select k(>=j) that maximizes |u_{ij}|
			for (uint32 j = i + 1; j < n; j++)
			{
				if (std::abs(U[j * rstep + i]) > std::abs(U[k * rstep + i])) k = j;
			}

			CHECK_GT(std::abs(U[k * rstep + i]), eps) << "major is zero";

			if (k != i)
			{
				// interchange rows of U: u(i,i:n)<-->u(k,i:n)
				for (uint32 j = i; j < n; j++)
				{
					std::swap(U[i * rstep + j], U[k * rstep + j]);
				}
				// interchange rows of P: p(i,:)<-->p(k,:)
				for (uint32 j = 0; j < n; j++)
				{
					std::swap(P[i * rstep + j], P[k * rstep + j]);
				}
				// interchange rows of L: l(i,0:i-1)<-->l(k,0:i-1)
				for (uint32 j = 0; j < i; j++)
				{
					std::swap(L[i * rstep + j], L[k * rstep + j]);
				}
			}

			float d = 1 / U[i * rstep + i];
			for (uint32 j = i + 1; j < n; j++)
			{
				float a = U[j * rstep + i] * d;
				L[j * rstep + i] = a;
				for (uint32 k = i; k < n; k++)
				{
					U[j * rstep + k] -= a * U[i * rstep + k];
				}
			}
		}
	}

	LUP::LUP() : Decomp(L"LU") {}

	void LUP::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const
	{
		const Tensor& A = bottom_blobs[0];
		//const Tensor& y = bottom_blobs[1];

		CHECK_EQ(A.shape[0], A.shape[1]);

		Tensor& L = top_blobs[0];
		Tensor& U = top_blobs[1];
		Tensor& P = top_blobs[2];

		L.Create(A.shape, A.shape.steps(), DataType::D4, Packing::CHW, opt.blob_allocator);
		U.Create(A.shape, A.shape.steps(), DataType::D4, Packing::CHW, opt.blob_allocator);
		P.Create(A.shape, A.shape.steps(), DataType::D4, Packing::CHW, opt.blob_allocator);

		LUImpl(A, P, L, U);
	}
}