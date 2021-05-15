#include "dnn/layers/binary_op.hpp"

namespace chaos
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
        inline float operator()(const float& x, const float& y) const { return std::pow(x, y); };
    };

    template<class Op>
    static inline void Operator(const Tensor& a, const Tensor& b, Tensor& c)
    {
        //Op op;
        Op op;
        size_t dims = c.shape.dims;
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

    template<class Op>
    static inline void Operator(const Tensor& a, float b, Tensor& c)
    {
        Op op;
        size_t dims = c.shape.dims;
        for (size_t i = 0; i < c.shape.total(); i++)
        {
            size_t a_idx = 0;
            size_t c_idx = 0;
            size_t idx = i;
            for (size_t d = 0; d < dims; d++)
            {
                size_t k = idx % c.shape[dims - d - 1];
                a_idx += (k >= a.shape[dims - d - 1] ? 0 : k) * a.steps[dims - d - 1];
                c_idx += k * c.steps[dims - d - 1];
                idx /= c.shape[dims - d - 1];
            }
            c[c_idx] = op(a[a_idx], b);
        }
    }

	BinaryOp::BinaryOp() : Layer("BinaryOp") {}

	void BinaryOp::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const
	{
        CHECK_EQ(2, bottom_blobs.size()) << "layer '" << type << "' expcec 2 inputs but got " << bottom_blobs.size();
        Tensor a = bottom_blobs[0];
        Tensor b = bottom_blobs[1];

        size_t a_dims = a.shape.dims;
        size_t b_dims = b.shape.dims;
        // reshape
        if (a_dims > b_dims)
        {
            b.shape.Insert(0, a_dims - b_dims, 1);
            b.steps.Insert(0, a_dims - b_dims, (uint32)b.shape.total());
        }
        else
        {
            a.shape.Insert(0, b_dims - a_dims, 1);
            a.steps.Insert(0, b_dims - a_dims, (uint32)a.shape.total());
        }

        CHECK_EQ(1, top_blobs.size()) << "layer '" << type << "' expcec 1 output but got " << top_blobs.size();
        Tensor& c = top_blobs[0];
        Shape c_shape = a.total() > b.total() ? a.shape : b.shape;
        if (c.empty()) c.Create(c_shape, c_shape.steps(), DataType::D4, Packing::CHW, opt.blob_allocator);
        CHECK_EQ(c_shape, c.shape);

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
        case MAX:
            Operator<BinaryMax>(a, b, c);
            break;
        case MIN:
            Operator<BinaryMin>(a, b, c);
            break;
        case POW:
            Operator<BinaryPow>(a, b, c);
            break;
        default:
            LOG(FATAL);
        }
	}

    void BinaryOp::Set(const std::string& pname, const std::any& val)
    {
        if ("op_type" == pname && val.type() == typeid(OpType))
        {
            op_type = std::any_cast<OpType>(val);
        }
        if ("f" == pname)
        {
            f = std::any_cast<float>(val);
        }
    }
}