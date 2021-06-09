#include "dnn/layers/svd.hpp"

//#include "utils/op.hpp"

namespace chaos::inline dnn
{
    void JacobiSVD(float* At, size_t astep, float* W_, float* Vt, size_t vstep,
        int m, int n, int n1, double minval = FLT_MIN, float eps = FLT_EPSILON * 2)
    {
        AutoBuffer<double> Wbuf(n);
        double* W = Wbuf.data();
        int i, j, k, iter, max_iter = std::max(m, 30);
        float c, s;
        double sd;

        for (i = 0; i < n; i++)
        {
            for (k = 0, sd = 0; k < m; k++)
            {
                float t = At[i * astep + k];
                sd += (double)t * t;
            }
            W[i] = sd;

            if (Vt)
            {
                for (k = 0; k < n; k++)
                    Vt[i * vstep + k] = 0;
                Vt[i * vstep + i] = 1;
            }
        }

        for (iter = 0; iter < max_iter; iter++)
        {
            bool changed = false;

            for (i = 0; i < n - 1; i++)
                for (j = i + 1; j < n; j++)
                {
                    float* Ai = At + i * astep, * Aj = At + j * astep;
                    double a = W[i], p = 0, b = W[j];

                    for (k = 0; k < m; k++)
                        p += (double)Ai[k] * Aj[k];

                    if (std::abs(p) <= eps * std::sqrt((double)a * b))
                        continue;

                    p *= 2;
                    double beta = a - b, gamma = hypot((double)p, beta);
                    if (beta < 0)
                    {
                        double delta = (gamma - beta) * 0.5;
                        s = (float)std::sqrt(delta / gamma);
                        c = (float)(p / (gamma * s * 2));
                    }
                    else
                    {
                        c = (float)std::sqrt((gamma + beta) / (gamma * 2));
                        s = (float)(p / (gamma * c * 2));
                    }

                    a = b = 0;
                    for (k = 0; k < m; k++)
                    {
                        float t0 = c * Ai[k] + s * Aj[k];
                        float t1 = -s * Ai[k] + c * Aj[k];
                        Ai[k] = t0; Aj[k] = t1;

                        a += (double)t0 * t0; b += (double)t1 * t1;
                    }
                    W[i] = a; W[j] = b;

                    changed = true;

                    if (Vt)
                    {
                        float* Vi = Vt + i * vstep, * Vj = Vt + j * vstep;
                        // givens
                        for (k = 0; k < n; k++)
                        {
                            float t0 = c * Vi[k] + s * Vj[k];
                            float t1 = -s * Vi[k] + c * Vj[k];
                            Vi[k] = t0; Vj[k] = t1;
                        }
                    }
                }
            if (!changed)
                break;
        }

        for (i = 0; i < n; i++)
        {
            for (k = 0, sd = 0; k < m; k++)
            {
                float t = At[i * astep + k];
                sd += (double)t * t;
            }
            W[i] = std::sqrt(sd);
        }

        for (i = 0; i < n - 1; i++)
        {
            j = i;
            for (k = i + 1; k < n; k++)
            {
                if (W[j] < W[k])
                    j = k;
            }
            if (i != j)
            {
                std::swap(W[i], W[j]);
                if (Vt)
                {
                    for (k = 0; k < m; k++)
                        std::swap(At[i * astep + k], At[j * astep + k]);

                    for (k = 0; k < n; k++)
                        std::swap(Vt[i * vstep + k], Vt[j * vstep + k]);
                }
            }
        }

        for (i = 0; i < n; i++)
            W_[i] = (float)W[i];

        if (!Vt)
            return;

        //RNG rng(0x12345678);
        uint64 state = 0x12345678;
        for (i = 0; i < n1; i++)
        {
            sd = i < n ? W[i] : 0;

            for (int ii = 0; ii < 100 && sd <= minval; ii++)
            {
                // if we got a zero singular value, then in order to get the corresponding left singular vector
                // we generate a random vector, project it to the previously computed left singular vectors,
                // subtract the projection and normalize the difference.
                const float val0 = (float)(1. / m);
                for (k = 0; k < m; k++)
                {
                    // RNG::next()
                    state = (uint64)(unsigned)state * /*CV_RNG_COEFF*/ 4164903690U + (unsigned)(state >> 32);
                    float val = ((unsigned)state & 256) != 0 ? val0 : -val0;
                    At[i * astep + k] = val;
                }
                for (iter = 0; iter < 2; iter++)
                {
                    for (j = 0; j < i; j++)
                    {
                        sd = 0;
                        for (k = 0; k < m; k++)
                            sd += At[i * astep + k] * At[j * astep + k];
                        float asum = 0;
                        for (k = 0; k < m; k++)
                        {
                            float t = (float)(At[i * astep + k] - sd * At[j * astep + k]);
                            At[i * astep + k] = t;
                            asum += std::abs(t);
                        }
                        asum = asum > eps * 100 ? 1 / asum : 0;
                        for (k = 0; k < m; k++)
                            At[i * astep + k] *= asum;
                    }
                }
                sd = 0;
                for (k = 0; k < m; k++)
                {
                    float t = At[i * astep + k];
                    sd += (double)t * t;
                }
                sd = std::sqrt(sd);
            }

            s = (float)(sd > minval ? 1 / sd : 0.);
            for (k = 0; k < m; k++)
                At[i * astep + k] *= s;
        }
    }

	SVD::SVD() : Layer("SVD") {}

    void SVD::CreatePipeline(const Option& opt)
    {
        transpose = LayerRegistry::CreateLayer("Permute");
        transpose->CreatePipeline(opt);
        transpose->Set("orders", Array<uint32>{1, 0});
    }

    void SVD::DestroyPipeline(const Option& opt) 
    {
        transpose->DestroyPipeline(opt);
    }

	void SVD::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const
	{
        CHECK_EQ(1, bottom_blobs.size()) << "expect 1 but got " << bottom_blobs.size() << " bottom blobs";
        CHECK_EQ(2, bottom_blobs[0].shape.size()) << "A must be a matrix";

        const Tensor& A = bottom_blobs[0];

        uint32 m = A.shape[0], n = A.shape[1];

        //CHECK_LE(1, top_blobs.size()) << "layer SVD expect at least 1 output but got " << top_blobs.size();
        //CHECK_GE(3, top_blobs.size()) << "layer SVD expect at most 3 output but got " << top_blobs.size();
        CHECK(1 == top_blobs.size() || 3 == top_blobs.size()) << "layer SVD expect 1 or 3 output(s) but got " << top_blobs.size();

        Tensor& W = top_blobs[0];

        bool compute_uv = top_blobs.size() == 3;
        bool full_uv = (uv & SVD::FULL_UV) != 0;

        if (uv & SVD::NO_UV)
        {
            compute_uv = full_uv = false;
        }

        bool at = false;
        if (m < n)
        {
            std::swap(m, n);
            at = true;
        }

        uint32 urows = full_uv ? m : n;
        uint32 astep = (uint32)AlignSize(m * 4ULL, 16) / 4;
        uint32 vstep = (uint32)AlignSize(n * 4ULL, 16) / 4;
        AutoBuffer<uchar> buf_((urows * astep + n * vstep + n) * 4ULL + 32); // urows * astep * esz + n * vstep * esz + n * esz + 32
        uchar* buf = AlignPtr(buf_.data(), 16);

        Tensor temp_a(Shape(n, m), Depth::D4, Packing::CHW, buf, Steps(astep, 1));
        Tensor temp_w(Shape((int)n), Depth::D4, Packing::CHW, buf + urows * astep * 4ULL);
        Tensor temp_u(Shape(urows, m), Depth::D4, Packing::CHW, buf, Steps(astep, 1));
        Tensor temp_v;
        if (compute_uv) temp_v = Tensor(Shape(n, n), Depth::D4, Packing::CHW, AlignPtr(buf + (urows * astep + n) * 4ULL, 16), Steps(vstep, 1));
        if (urows > n) memset(temp_u.data, 0, urows * m * 4ULL);

        if (not at)
        {
            //transpose(A, temp_a);
            std::vector<Tensor> tops{temp_a};
            transpose->Forward({A}, tops, opt);
        }
        else
        {
            A.CopyTo(temp_a);
        }

        JacobiSVD((float*)temp_a.data, astep, (float*)temp_w.data, (float*)temp_v.data, vstep, m, n, compute_uv ? urows : 0);

        if (W.empty()) W.Create(Shape(n), Steps(1), Depth::D4, Packing::CHW, opt.blob_allocator);
        temp_w.CopyTo(W);
        if (compute_uv)
        {
            Tensor& U = top_blobs[1];
            Tensor& Vt = top_blobs[2];
            if (!at)
            {
                if (U.empty()) U.Create(Shape(m, urows), Steps(urows, 1), Depth::D4, Packing::CHW, opt.blob_allocator);
                CHECK_EQ(Shape(m, urows), U.shape);
                if (Vt.empty()) Vt.Create(Shape(n, n), Steps(n,1), Depth::D4, Packing::CHW, opt.blob_allocator);
                CHECK_EQ(Shape(n, n), Vt.shape);
                std::vector<Tensor> tops{U};
                transpose->Forward({ temp_u }, tops);
                //transpose(temp_u, U);
                temp_v.CopyTo(Vt);
            }
            else
            {
                if (U.empty()) U.Create(Shape(n, n), Steps(n,1), Depth::D4, Packing::CHW, opt.blob_allocator);
                CHECK_EQ(Shape(n, n), U.shape);
                if (Vt.empty()) Vt.Create(Shape(urows, m), Steps(m,1), Depth::D4, Packing::CHW, opt.blob_allocator);
                CHECK_EQ(Shape(urows, m), Vt.shape);
                std::vector<Tensor> tops{U};
                transpose->Forward({temp_v}, tops);
                //transpose(temp_v, U);
                temp_u.CopyTo(Vt);
            }
        }
	}

    void SVD::Set(const std::string& pname, const std::any& param)
    {
        if ("uv" == pname) uv = std::any_cast<UV>(param);
    }
}
