#include "dnn/layers/invert.hpp"
#include "utils/op.hpp"

namespace chaos
{
	Invert::Invert() : Layer("Invert") {}

	void Invert::Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const
	{
		CHECK_EQ(2, bottom_blob.shape.dims);

		uint32 m = bottom_blob.shape[0];
		uint32 n = bottom_blob.shape[1];

        if (method == Decomp::SVD)
        {

        }

        if (method == Decomp::EIG)
        {
            //Tensor a;
            //Operator::Permute(a, /*orders=*/{ 1u, 0u }); // transpose
        }

		CHECK_EQ(m, n);
		top_blob.Create(Shape(n, n), /*steps=*/{n, 1u}, DataType::D4, Packing::CHW, opt.blob_allocator);

		if (n <= 3)
		{
            const float* sdata = (const float*)bottom_blob.data;
            float* ddata = (float*)top_blob.data;
            uint32 sstep = bottom_blob.steps[0];
            uint32 dstep = top_blob.steps[0];

            auto sf = [=](int y, int x)->const float& { return (sdata + y * sstep)[x]; };
            auto df = [=](int y, int x)->float& { return (ddata + y * dstep)[x]; };
            if (n == 2)
            {
                // det2
                float d = sf(0, 0) * sf(1, 1) - sf(0, 1) * sf(1, 0);
                CHECK_GT(std::abs(d), eps) << "the matrix is singular";
                d = 1.f / d;
                df(1,1) = sf(0, 0) * d;
                df(0,0) = sf(1, 1) * d;
                df(0,1) = -sf(0, 1) * d;
                df(1,0) = -sf(1, 0) * d;

            }
            else if (n == 3)
            {
                // det3
                float d =
                    sf(0, 0) * (sf(1, 1) * sf(2, 2) - sf(1, 2) * sf(2, 1)) -
                    sf(0, 1) * (sf(1, 0) * sf(2, 2) - sf(1, 2) * sf(2, 0)) +
                    sf(0, 2) * (sf(1, 0) * sf(2, 1) - sf(1, 1) * sf(2, 0));

                CHECK_GT(std::abs(d), eps) << "the matrix is singular";

                d = 1.f / d;
                df(0, 0) = (sf(1, 1) * sf(2, 2) - sf(1, 2) * sf(2, 1)) * d;
                df(0, 1) = (sf(0, 2) * sf(2, 1) - sf(0, 1) * sf(2, 2)) * d;
                df(0, 2) = (sf(0, 1) * sf(1, 2) - sf(0, 2) * sf(1, 1)) * d;

                df(1, 0) = (sf(1, 2) * sf(2, 0) - sf(1, 0) * sf(2, 2)) * d;
                df(1, 1) = (sf(0, 0) * sf(2, 2) - sf(0, 2) * sf(2, 0)) * d;
                df(1, 2) = (sf(0, 2) * sf(1, 0) - sf(0, 0) * sf(1, 2)) * d;

                df(2, 0) = (sf(1, 0) * sf(2, 1) - sf(1, 1) * sf(2, 0)) * d;
                df(2, 1) = (sf(0, 1) * sf(2, 0) - sf(0, 0) * sf(2, 1)) * d;
                df(2, 2) = (sf(0, 0) * sf(1, 1) - sf(0, 1) * sf(1, 0)) * d;
            }
            else // n == 1
            {
                CHECK_EQ(n, 1);
                CHECK_GT(std::abs(sdata[0]), eps);
                ddata[0] = 1.f / sdata[0];
            }
		}

        if (method == Decomp::LUP)
        {

        }
	}
}