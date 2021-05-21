#pragma once

#include "core/core.hpp"
#include "dnn/layer.hpp"

namespace chaos
{
    template<class Op>
    class Operator
    {
    public:
        static Op& GetInstance()
        {
            static Op op;
            return op;
        }

        Tensor operator()(const Tensor& a, const Tensor& b) const
        {
            std::vector<Tensor> tops(1);
            layer->Forward({a,b}, tops);
            return tops[0];
        }

        Operator(const Operator&) = delete;
        Operator& operator=(const Operator&) = delete;
    public:
        Operator() = default;
        virtual ~Operator() = default;
        Ptr<dnn::Layer> layer;
    };

    class CHAOS_API Add : public Operator<Add>
    {
    public:
        Add();
        
        Add(const Add&) = delete;
        Add& operator=(const Add&) = delete;
    };

    class CHAOS_API Div : public Operator<Div>
    {
    public:
        Div();

        Div(const Div&) = delete;
        Div& operator=(const Div&) = delete;
    };

    class CHAOS_API Dot : public Operator<Dot>
    {
    public:
        Dot();

        Dot(const Dot&) = delete;
        Dot& operator=(const Dot&) = delete;
    };
    class CHAOS_API GEMM : public Operator<GEMM>
    {
    public:
        enum
        {
            NOTRANS = 0,
            TRANSA = 1,
            TRANSB = 2,
            TRANSC = 4,
        };

        GEMM();
        Tensor operator()(const Tensor& a, const Tensor& b);
        Tensor operator()(int flag, const Tensor& a, const Tensor& b, float alpha, const Tensor& c, float beta) const;

        GEMM(const GEMM&) = delete;
        GEMM& operator=(const GEMM&) = delete;
    };
    class CHAOS_API Mul : public Operator<Mul>
    {
    public:
        Mul();

        Mul(const Mul&) = delete;
        Mul& operator=(const Mul&) = delete;
    };

    class CHAOS_API Sub : public Operator<Sub>
    {
    public:
        Sub();
        
        Sub(const Sub&) = delete;
        Sub& operator=(const Sub&) = delete;
    };


    template<class...Shapes>
    Shape BroadcastShapes(const Shapes&...shapes)
    {
        std::vector<Shape> shapes_;
        int dims = 0;
        auto Cast = [&](const Shape& shape) {
            shapes_.push_back(shape);
            dims = std::max(dims, (int)shape.dims);
        };
        (..., Cast(shapes));

        Shape broadcast = std::vector<uint32>(dims, 1);
        for (int i = 0; i < dims; i++)
        {
            for (const auto& shape : shapes_)
            {
                int offset = shape.dims - dims;
                if (i + offset >= 0)
                {
                    if (broadcast[i] == 1)
                    {
                        broadcast[i] = shape[i + offset];
                    }
                    else // new_shape[i] != 1
                    {
                        if (shape[i + offset] != 1) CHECK_EQ(broadcast[i], shape[i + offset]) << "can not broadcast";
                    }
                }
            }
        }
        return broadcast;
    }

	Tensor operator+(const Tensor& a, const Tensor& b);
    Tensor operator+(float a, const Tensor& b);
    Tensor operator+(const Tensor& a, float b);

    Tensor operator-(const Tensor& a, const Tensor& b);
    Tensor operator-(float a, const Tensor& b);
    Tensor operator-(const Tensor& a, float b);

    Tensor operator*(const Tensor& a, const Tensor& b);
    Tensor operator*(float a, const Tensor& b);
    Tensor operator*(const Tensor& a, float b);

    Tensor operator/(const Tensor& a, const Tensor& b);
    Tensor operator/(float a, const Tensor& b);
    Tensor operator/(const Tensor& a, float b);

    Tensor cross(const Tensor& a, const Tensor& b);
    Tensor dot(const Tensor& a, const Tensor& b);
    Tensor mean(const Tensor& a);
    Tensor mean(const Tensor& a, const std::vector<uint32>& vecdim);
    Tensor mul(const Tensor& a, const Tensor& b);
    Tensor sum(const Tensor& a);
    Tensor sum(const Tensor& a, const std::vector<uint32>& vecdim);

}