#include "dnn/layers/gemm.hpp"

namespace chaos
{
	GEMM::GEMM() : Layer(L"GEMM") {}

	void GEMM::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const
	{
		const Tensor& A = bottom_blobs[0];
		const Tensor& B = bottom_blobs[1];
		//const Tensor& C = bottom_blobs[2];

		CHECK_EQ(2, A.shape.dims);
		CHECK_EQ(2, B.shape.dims);
		CHECK_EQ(A.shape[1], B.shape[0]);

		uint32 m = A.shape[0];
		uint32 n = A.shape[1];
		uint32 k = B.shape[1];

		Tensor& C = top_blobs[0];
		C.Create(Shape(m, k), { k, 1u }, DataType::D4, Packing::CHW, opt.blob_allocator);
		memset(C.data, 0, m * k * sizeof(float));

		uint32 astep = A.steps[0];
		uint32 bstep = B.steps[0];

		for (size_t r = 0; r < n; r++)
		{
			for (size_t c = 0; c < k; c++)
			{
				float* row = (float*)A.data + r * astep;
				float* col = (float*)B.data + c;
				for (size_t i = 0; i < n; i++)
				{
					C[r * k + c] += row[i] * col[i * bstep];
				}
			}
		}
	}
}