#include "dnn/layers/invert.hpp"
#include "dnn/layers/svd.hpp"
#include "dnn/layers/backsubst.hpp"

namespace chaos
{
	inline namespace dnn
	{
		Invert::Invert() : Layer("Invert") 
		{
			svd = std::make_shared<SVD>();
			backsubst = std::make_shared<Backsubst>();
		}

		void Invert::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const
		{
            CHECK_EQ(1, bottom_blobs.size());
            CHECK_EQ(1, top_blobs.size());

            const Tensor& A = bottom_blobs[0];

            int m = A.shape[0], n = A.shape[1];

            if (method == DECOMP_SVD || m != n) // SVD
            {
                std::vector<Tensor> wuvt(3);
                svd->Forward(bottom_blobs, wuvt, opt);
                backsubst->Forward(wuvt, top_blobs, opt);
            }
            else // m == n
            {
                Tensor& At = top_blobs[0];
                if (At.empty()) At.Create(Shape(n, n), Steps(n, 1), Depth::D4, Packing::CHW, opt.blob_allocator);
                CHECK_EQ(Shape(n, n), At.shape);

                if (n <= 3)
                {
                    auto Sf = [&](int y, int x)->const float& { return ((const float*)A.data + y * n)[x]; };
                    auto Df = [&](int y, int x)->float& { return ((float*)At.data + y * n)[x]; };

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
                //else // use LUP etc
                //{
                //}
            }
		}

        void Invert::Set(const std::string& pname, const std::any& param)
        {
            if ("method" == pname) method = std::any_cast<Method>(param);
        }
	}
}