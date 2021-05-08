#include "dnn/layers/gemm.hpp"

namespace chaos
{
	GEMM::GEMM() : Layer("GEMM") {}

	void GEMM::Set(const std::string& name, const std::any& val)
	{
		if ("alpha" == name)
		{
			alpha = std::any_cast<float>(val);
		}
		if ("beta" == name)
		{
			beta = std::any_cast<float>(val);
		}
	}

	void GEMM::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const
	{
		CHECK_LE(2, bottom_blobs.size()) << "layer '" << type << "' expect 2 inputs at least but got " << bottom_blobs.size();
		CHECK_GE(3, bottom_blobs.size()) << "layer '" << type << "' expect 3 inputs at most but got " << bottom_blobs.size();
		const Tensor& A = bottom_blobs[0];
		const Tensor& B = bottom_blobs[1];
		//const Tensor& C = bottom_blobs.size() > 2 ? bottom_blobs[2] : Tensor();

		CHECK_EQ(2, A.shape.dims) << "input A must be a matrix";
		CHECK_EQ(2, B.shape.dims) << "input B must be a matrix";
		CHECK_EQ(A.shape[1], B.shape[0]) << "cols of A shoule be same with rows of B";

		uint32 m = A.shape[0];
		uint32 n = A.shape[1];
		uint32 k = B.shape[1];

		CHECK_EQ(1, top_blobs.size()) << "layer '" << type << "' expect 1 output but got " << top_blobs.size();
		Tensor& C = top_blobs[0];
		if (C.empty()) C.Create(Shape(m, k), Steps{ k, 1u }, DataType::D4, Packing::CHW, opt.blob_allocator);
		if (bottom_blobs.size() == 2 || bottom_blobs[2].empty())
		{
			memset(C.data, 0, sizeof(float) * C.total());
		}
		else
		{
			CHECK_EQ(2, bottom_blobs[2].shape.dims) << "input C must be a matrix";
			CHECK_EQ(bottom_blobs[2].shape, C.shape) << "shape of input C shoule be " << m << "x" << k;
			bottom_blobs[2].CopyTo(C);
		}

		uint32 astep = A.steps[0];
		uint32 bstep = B.steps[0];
		uint32 cstep = C.steps[0];
		for (size_t r = 0; r < m; r++)
		{
			for (size_t c = 0; c < k; c++)
			{
				float* row = (float*)A.data + r * astep;
				float* col = (float*)B.data + c;

				C[r * k + c] *= beta;
				for (size_t i = 0; i < n; i++)
				{
					C[r * cstep + c] += (alpha * row[i] * col[i * bstep]);
				}
			}
		}
	}
}