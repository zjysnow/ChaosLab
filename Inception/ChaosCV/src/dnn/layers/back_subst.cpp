#include "dnn/layers/back_subst.hpp"

namespace chaos
{
    inline namespace dnn
    {
        template<typename T1, typename T2, typename T3>
        static void MatrAXPY(int m, int n, const T1* x, int dx,
            const T2* a, int inca, T3* y, int dy)
        {
            for (int i = 0; i < m; i++, x += dx, y += dy)
            {
                T2 s = a[i * inca];
                for (int j = 0; j < n; j++)
                    y[j] = (T3)(y[j] + s * x[j]);
            }
        }


        static void SVBkSb(int m, int n, const float* w, int incw,
            const float* u, int ldu, bool uT,
            const float* v, int ldv, bool vT,
            const float* b, int ldb, int nb,
            float* x, int ldx, double* buffer)
        {
            double threshold = 0;
            int udelta0 = uT ? ldu : 1, udelta1 = uT ? 1 : ldu;
            int vdelta0 = vT ? ldv : 1, vdelta1 = vT ? 1 : ldv;
            int i, j, nm = std::min(m, n);

            if (!b)
                nb = m;

            for (i = 0; i < n; i++)
                for (j = 0; j < nb; j++)
                    x[i * ldx + j] = 0;

            for (i = 0; i < nm; i++)
                threshold += w[i * incw];
            threshold *= (float)(DBL_EPSILON * 2);

            // v * inv(w) * uT * b
            for (i = 0; i < nm; i++, u += udelta0, v += vdelta0)
            {
                double wi = w[i * incw];
                if ((double)std::abs(wi) <= threshold)
                    continue;
                wi = 1 / wi;

                if (nb == 1)
                {
                    double s = 0;
                    if (b)
                        for (j = 0; j < m; j++)
                            s += u[j * udelta1] * b[j * ldb];
                    else
                        s = u[0];
                    s *= wi;

                    for (j = 0; j < n; j++)
                        x[j * ldx] = (float)(x[j * ldx] + s * v[j * vdelta1]);
                }
                else
                {
                    if (b)
                    {
                        for (j = 0; j < nb; j++)
                            buffer[j] = 0;
                        MatrAXPY(m, nb, b, ldb, u, udelta1, buffer, 0);

                        for (j = 0; j < nb; j++)
                            buffer[j] *= wi;
                    }
                    else
                    {
                        for (j = 0; j < nb; j++)
                            buffer[j] = u[j * udelta1] * wi;
                    }
                    MatrAXPY(n, nb, buffer, 0, v, vdelta1, x, ldx);
                }
            }
        }

        BackSubst::BackSubst() : Layer("BackSubst") {}

        void BackSubst::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const
        {
            CHECK_LE(3, bottom_blobs.size()) << "layer backsubst expect at least 3 inputs but got " << bottom_blobs.size();
            CHECK_GE(4, bottom_blobs.size()) << "layer backsubst expect at most 4 inputs but got " << bottom_blobs.size();
            const Tensor& w = bottom_blobs[0];
            const Tensor& u = bottom_blobs[1];
            const Tensor& vt = bottom_blobs[2];
            Tensor rhs = bottom_blobs.size() == 4 ? bottom_blobs[3] : Tensor();

            //CHECK_EQ(1, w.shape.size());
            CHECK(1 == w.shape.size() || 2 == w.shape.size());
            CHECK_EQ(2, u.shape.size());
            CHECK_EQ(2, vt.shape.size());

            int esz = 1 * w.depth * w.packing;
            auto depth = w.depth;
            auto packing = w.packing;

            uint32 m = u.shape[0];
            uint32 n = vt.shape[1];
            uint32 nb = rhs.empty() ? m : rhs.shape[1];
            uint32 nm = std::min(m, n);
            size_t wstep = w.shape.size() == 1 ? 1ULL : (w.shape[0] == 1 ? 1ULL : w.shape[1] == 1 ? 1 : w.steps[0] + 1); // maybe
            AutoBuffer<uchar> buffer(nb * sizeof(double) + 16);

            CHECK(w.depth == u.depth && u.depth == vt.depth && u.data && vt.data && w.data);
            CHECK(u.shape[1] >= nm && vt.shape[0] >= nm &&
                (w.shape == Shape(nm) || w.shape == Shape(nm, 1) || w.shape == Shape(1, nm) || w.shape == Shape(vt.shape[0], u.shape[1])));
            CHECK(rhs.data == 0 || (rhs.depth == depth && rhs.shape[0] == m));

            CHECK_EQ(1, top_blobs.size()) << "layer backsubst expect 1 output but got " << top_blobs.size();
            Tensor& dst = top_blobs[0];
            if (dst.empty()) dst.Create(Shape(n, nb), Steps(nb, 1), depth, packing, opt.blob_allocator);
            CHECK_EQ(Shape(n, nb), dst.shape);

            SVBkSb(m, n, (const float*)w.data, (int)wstep,
                (const float*)u.data, u.steps[0], false,
                (const float*)vt.data, vt.steps[0], true,
                (const float*)rhs.data, rhs.empty() ? 0 : rhs.steps[0], nb,
                (float*)dst.data, dst.steps[0],
                (double*)AlignPtr(buffer.data(), sizeof(double)));
        }
    }
}