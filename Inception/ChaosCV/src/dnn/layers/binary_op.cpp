#include "dnn/layers/binary_op.hpp"

namespace chaos
{
    inline namespace dnn
    {
        struct BinaryAdd
        {
            inline float operator()(const float& x, const float& y) const { return x + y; }
        };

        struct BinarySub
        {
            inline float operator()(const float& x, const float& y) const { return x - y; }
        };

        struct BinaryMul
        {
            inline float operator()(const float& x, const float& y) const { return x * y; }
        };

        struct BinaryDiv
        {
            inline float operator()(const float& x, const float& y) const { return x / y; }
        };

        struct BinaryMax
        {
            inline float operator()(const float& x, const float& y) const { return std::max(x, y); }
        };

        struct BinaryMin
        {
            inline float operator()(const float& x, const float& y) const { return std::min(x, y); }
        };

        struct BinaryPow
        {
            inline float operator()(const float& x, const float& y) const { return std::powf(x, y); }
        };

        struct BinarySub2
        {
            inline float operator()(const float& x, const float& y) const { return (x - y) * (x - y); }
        };
        
        template<class Op>
        static inline void Operator(const Tensor& a, const Tensor& b, Tensor& c)
        {
            Op op;
            size_t dims = c.shape.size();
            for (size_t i = 0; i < c.shape.total(); i++)
            {
                size_t a_idx = 0;
                size_t b_idx = 0;
                size_t c_idx = 0;
                size_t idx = i;
                for (size_t d = 0; d < dims; d++)
                {
                    size_t k = idx % c.shape[dims - d - 1];
                    a_idx += (k >= a.shape[dims - d - 1] ? 0 : k) * a.steps[dims - d - 1];
                    b_idx += (k >= b.shape[dims - d - 1] ? 0 : k) * b.steps[dims - d - 1];
                    c_idx += k * c.steps[dims - d - 1];
                    idx /= c.shape[dims - d - 1];
                }
                c[c_idx] = op(a[a_idx], b[b_idx]);
            }
        }

        BinaryOp::BinaryOp() : Layer("BinaryOp") {}
        BinaryOp::BinaryOp(int type) : Layer("BinaryOp") { op_type = type; }

        void BinaryOp::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const
        {
            CHECK_EQ(2, bottom_blobs.size()) << "layer 'BinaryOp' expcet 2 inputs but got " << bottom_blobs.size();
            Tensor a = bottom_blobs[0];
            Tensor b = bottom_blobs[1];

            int a_dims = (int)a.shape.size();
            int b_dims = (int)b.shape.size();
            // expand the shape to the same dims
            int a_cnt = std::max(0, b_dims - a_dims);
            int b_cnt = std::max(0, a_dims - b_dims);
            a.steps.Expand(0, a_cnt, (uint32)a.total());
            a.shape.Expand(0, a_cnt);
            b.steps.Expand(0, b_cnt, (uint32)b.total());
            b.shape.Expand(0, b_cnt);

            CHECK_EQ(1, top_blobs.size()) << "layer 'BinaryOp' expcet 1 output but got " << top_blobs.size();
            Tensor& c = top_blobs[0];
            Shape c_shape = a.shape & b.shape; //Broadcast(a.shape, b.shape);
            if (c.empty()) c.Create(c_shape, c_shape.steps(), Depth::D4, Packing::CHW, opt.blob_allocator);
            CHECK_EQ(c_shape, c.shape) << "expect " << c_shape << " but got " << c.shape;

            switch (op_type)
            {
            case ADD:
                Operator<BinaryAdd>(a, b, c);
                break;
            case SUB:
                Operator<BinarySub>(a, b, c);
                break;
            case MUL:
                Operator<BinaryMul>(a, b, c);
                break;
            case DIV:
                Operator<BinaryDiv>(a, b, c);
                break;
            case MIN:
                Operator<BinaryMin>(a, b, c);
                break;
            case MAX:
                Operator<BinaryMax>(a, b, c);
                break;
            case POW:
                Operator<BinaryPow>(a, b, c);
                break;
            case SUB2:
                Operator<BinarySub2>(a, b, c);
                break;
            default:
                LOG(FATAL) << "unsupport binary method";
                break;
            }
        }

        void BinaryOp::Set(const std::string& pname, const std::any& param)
        {
            if ("op_type" == pname) op_type = std::any_cast<OpType>(param);
        }
    }
}