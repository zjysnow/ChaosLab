#include "dnn/layers/gemm.hpp"

namespace chaos
{
	inline namespace dnn
	{
		GEMM::GEMM() : Layer("GEMM") {}

		void GEMM::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const
		{
			CHECK_LE(2, bottom_blobs.size()) << "layer '" << type << "' expect 2 inputs at least but got " << bottom_blobs.size();
			CHECK_GE(3, bottom_blobs.size()) << "layer '" << type << "' expect 3 inputs at most but got " << bottom_blobs.size();
			const Tensor& A = bottom_blobs[0];
			const Tensor& B = bottom_blobs[1];

			CHECK_EQ(2, A.shape.size()) << "input A must be a matrix";
			CHECK_EQ(2, B.shape.size()) << "input B must be a matrix";

			uint32 m = A.shape[0];
			uint32 n = A.shape[1];
			uint32 p = B.shape[0];
			uint32 k = B.shape[1];

			if (transA) std::swap(m, n);
			if (transB) std::swap(p, k);

			CHECK_EQ(n, p) << "cols of A shoule be same with rows of B";

			CHECK_EQ(1, top_blobs.size()) << "layer '" << type << "' expect 1 output but got " << top_blobs.size();
			Tensor& C = top_blobs[0];
			if (C.empty()) C.Create(Shape(m, k), Steps(k, 1), Depth::D4, Packing::CHW, opt.blob_allocator);
			CHECK_EQ(Shape(m, k), C.shape);

			uint32 astep = A.steps[0];
			uint32 bstep = B.steps[0];
			uint32 cstep = C.steps[0];

			if (bottom_blobs.size() == 2 || beta == 0)
			{
				memset(C.data, 0, C.total() * C.depth * C.packing);
			}
			else
			{
				CHECK(not bottom_blobs[2].empty()) << "really?";
				CHECK_EQ(2, bottom_blobs[2].shape.size()) << "input C must be a matrix";
				CHECK_EQ(bottom_blobs[2].shape, C.shape) << "shape of input C shoule be " << m << "x" << k;
				bottom_blobs[2].CopyTo(C);
			}

			for (size_t r = 0; r < m; r++)
			{
				for (size_t c = 0; c < k; c++)
				{
					C[r * cstep + c] *= beta;
					for (size_t i = 0; i < n; i++)
					{
						float a = transA ? A[r + i * astep] : A[i + r * astep];
						float b = transB ? B[i + c * bstep] : B[c + i * bstep];
						C[r * cstep + c] += (alpha * a * b);
					}
				}
			}
		}

		void GEMM::Set(const std::string& pname, const std::any& param)
		{
			if ("transA" == pname) transA = std::any_cast<bool>(param);
			if ("transB" == pname) transB = std::any_cast<bool>(param);
			if ("alpha" == pname) alpha = std::any_cast<float>(param);
			if ("beta" == pname) beta = std::any_cast<float>(param);
		}
	}
}