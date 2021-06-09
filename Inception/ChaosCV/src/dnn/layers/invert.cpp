#include "dnn/layers/invert.hpp"

namespace chaos::inline dnn
{
	Invert::Invert() : Layer("Invert") {}

    void Invert::CreatePipeline(const Option& opt)
    {
        svd = LayerRegistry::CreateLayer("SVD");
        svd->CreatePipeline(opt);

        backsubst = LayerRegistry::CreateLayer("BackSubst");
        backsubst->CreatePipeline(opt);
    }
    void Invert::DestroyPipeline(const Option& opt)
    {
        svd->DestroyPipeline(opt);
        backsubst->DestroyPipeline(opt);
    }
    
	void Invert::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const
	{
		CHECK_EQ(1, bottom_blobs.size());
		CHECK_EQ(1, top_blobs.size());

		const Tensor& src = bottom_blobs[0];
		Tensor& dst = top_blobs[0];
		uint32 m = src.shape[0], n = src.shape[1];
        auto depth = src.depth;
        auto packing = src.packing;
        size_t esz = 1 * src.depth * src.packing;

		if (m != n) // SVD
		{
            int nm = std::min(m, n);
            AutoBuffer<uchar> buf_((m * nm + nm + nm * n) * esz + sizeof(double));
            uchar* buf = AlignPtr((uchar*)buf_.data(), (int)esz);
            std::vector<Tensor> wuvt(3);
            wuvt[1] = Tensor(Shape(m, nm), depth, packing, buf); // u
            wuvt[0] = Tensor(Shape(nm), depth, packing, buf + esz * m * nm); // w
            wuvt[2] = Tensor(Shape(nm, n), depth, packing, buf + (m + 1) * nm * esz); // vt
            svd->Forward({ src }, wuvt, opt);
            backsubst->Forward(wuvt, top_blobs, opt);
		}
		else
		{
            if (dst.empty()) dst.Create(Shape(n,n), Steps(n,1), Depth::D4, Packing::CHW, opt.blob_allocator);
            CHECK_EQ(Shape(n,n), dst.shape);
			if (n <= 3)
			{
                const uchar* srcdata = (const uchar*)src.data;
                uchar* dstdata = (uchar*)dst.data;
                size_t srcstep = src.steps[0] * src.depth * src.packing;
                size_t dststep = dst.steps[0] * src.depth * src.packing;

                auto Sf = [=](int y, int x)->const float& { return ((float*)(srcdata + y * srcstep))[x]; };
                auto Df = [=](int y, int x)->float& { return ((float*)(dstdata + y * dststep))[x]; };

                if (n == 2)
                {
                    // det2
                    double d = (double)Sf(0, 0) * Sf(1, 1) - (double)Sf(0, 1) * Sf(1, 0);
                    if (d != 0.)
                    {
                        d = 1. / d;
                        double t0, t1;
                        t0 = Sf(0, 0) * d;
                        t1 = Sf(1, 1) * d;
                        Df(1, 1) = (float)t0;
                        Df(0, 0) = (float)t1;
                        t0 = -Sf(0, 1) * d;
                        t1 = -Sf(1, 0) * d;
                        Df(0, 1) = (float)t0;
                        Df(1, 0) = (float)t1;
                    }
                }
                else if (n == 3)
                {
                    // det3
                    double d =
                        Sf(0, 0) * ((double)Sf(1, 1) * Sf(2, 2) - (double)Sf(1, 2) * Sf(2, 1)) -
                        Sf(0, 1) * ((double)Sf(1, 0) * Sf(2, 2) - (double)Sf(1, 2) * Sf(2, 0)) +
                        Sf(0, 2) * ((double)Sf(1, 0) * Sf(2, 1) - (double)Sf(1, 1) * Sf(2, 0));

                    if (d != 0.)
                    {
                        double t[12];

                        //result = true;
                        d = 1. / d;
                        t[0] = (((double)Sf(1, 1) * Sf(2, 2) - (double)Sf(1, 2) * Sf(2, 1)) * d);
                        t[1] = (((double)Sf(0, 2) * Sf(2, 1) - (double)Sf(0, 1) * Sf(2, 2)) * d);
                        t[2] = (((double)Sf(0, 1) * Sf(1, 2) - (double)Sf(0, 2) * Sf(1, 1)) * d);

                        t[3] = (((double)Sf(1, 2) * Sf(2, 0) - (double)Sf(1, 0) * Sf(2, 2)) * d);
                        t[4] = (((double)Sf(0, 0) * Sf(2, 2) - (double)Sf(0, 2) * Sf(2, 0)) * d);
                        t[5] = (((double)Sf(0, 2) * Sf(1, 0) - (double)Sf(0, 0) * Sf(1, 2)) * d);

                        t[6] = (((double)Sf(1, 0) * Sf(2, 1) - (double)Sf(1, 1) * Sf(2, 0)) * d);
                        t[7] = (((double)Sf(0, 1) * Sf(2, 0) - (double)Sf(0, 0) * Sf(2, 1)) * d);
                        t[8] = (((double)Sf(0, 0) * Sf(1, 1) - (double)Sf(0, 1) * Sf(1, 0)) * d);

                        Df(0, 0) = (float)t[0]; Df(0, 1) = (float)t[1]; Df(0, 2) = (float)t[2];
                        Df(1, 0) = (float)t[3]; Df(1, 1) = (float)t[4]; Df(1, 2) = (float)t[5];
                        Df(2, 0) = (float)t[6]; Df(2, 1) = (float)t[7]; Df(2, 2) = (float)t[8];
                    }
                }
                else
                {
                    CHECK_EQ(n, 1);
                    double d = Sf(0, 0);
                    if (d != 0.)
                    {
                        //result = true;
                        Df(0, 0) = (float)(1. / d);
                    }
                }
			}
			else // use LUP etc
			{

			}
		}
	}
}