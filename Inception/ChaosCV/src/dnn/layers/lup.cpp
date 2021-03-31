#include "dnn/layers/lup.hpp"

namespace chaos
{
	LUP::LUP() : Decomp(L"LUP") {}

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

		L.Create(A.shape, A.shape.steps(), DataType::D4, Packing::CHW, opt.blob_allocator);
		U.Create(A.shape, A.shape.steps(), DataType::D4, Packing::CHW, opt.blob_allocator);
		P.Create(A.shape, A.shape.steps(), DataType::D4, Packing::CHW, opt.blob_allocator);

		uint32 n = A.shape[0];

		// to preset U, L, P
		A.CopyTo(U);
		memset(L.data, 0, sizeof(float) * n * n);
		memset(P.data, 0, sizeof(float) * n * n);
		for (size_t i = 0; i < n; i++)
		{
			L[i * n + i] = P[i * n + i] = 1.f;
		}

		uint32 rstep = U.steps[0];
		size_t k;
		for (size_t i = 0; i < n - 1; i++)
		{
			k = i;
			// select k(>=j) that maximizes |u_{ij}|
			for (size_t j = i + 1; j < n; j++)
			{
				if (std::abs(U[j * rstep + i]) > std::abs(U[k * rstep + i])) k = j;
			}

			CHECK_GT(std::abs(U[k * rstep + i]), 1e-5) << "major is zero";

			if (k != i)
			{
				// interchange rows of U: u(i,i:n)<-->u(k,i:n)
				for (size_t j = i; j < n; j++)
				{
					std::swap(U[i * rstep + j], U[k * rstep + j]);
				}
				// interchange rows of P: p(i,:)<-->p(k,:)
				for (size_t j = 0; j < n; j++)
				{
					std::swap(P[i * rstep + j], P[k * rstep + j]);
				}
				// interchange rows of L: l(i,0:i-1)<-->l(k,0:i-1)
				for (size_t j = 0; j < i; j++)
				{
					std::swap(L[i * rstep + j], L[k * rstep + j]);
				}
			}

			float d = 1 / U[i * rstep + i];
			for (size_t j = i + 1; j < n; j++)
			{
				float a = U[j * rstep + i] * d;
				L[j * rstep + i] = a; // L_{ji} = U_{ji} / U_{ii}
				for (size_t k = i; k < n; k++)
				{
					U[j * rstep + k] -= a * U[i * rstep + k];
				}
			}
		}
	}
}