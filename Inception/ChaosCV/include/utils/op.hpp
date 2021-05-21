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

        Operator(const Operator&) = delete;
        Operator& operator=(const Operator&) = delete;
    public:
        Operator() = default;
        virtual ~Operator() = default;
        Ptr<dnn::Layer> layer;
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

	CHAOS_API Tensor operator+(const Tensor& a, const Tensor& b);
    CHAOS_API Tensor operator+(float a, const Tensor& b);
    CHAOS_API Tensor operator+(const Tensor& a, float b);

    CHAOS_API Tensor operator-(const Tensor& a, const Tensor& b);
    CHAOS_API Tensor operator-(float a, const Tensor& b);
    CHAOS_API Tensor operator-(const Tensor& a, float b);

    CHAOS_API Tensor operator*(const Tensor& a, const Tensor& b);
    CHAOS_API Tensor operator*(float a, const Tensor& b);
    CHAOS_API Tensor operator*(const Tensor& a, float b);

    CHAOS_API Tensor operator/(const Tensor& a, const Tensor& b);
    CHAOS_API Tensor operator/(float a, const Tensor& b);
    CHAOS_API Tensor operator/(const Tensor& a, float b);

    CHAOS_API Tensor cross(const Tensor& a, const Tensor& b);
    CHAOS_API Tensor dot(const Tensor& a, const Tensor& b);
    CHAOS_API Tensor mean(const Tensor& a);
    CHAOS_API Tensor mean(const Tensor& a, const std::vector<uint32>& vecdim);
    CHAOS_API Tensor mul(const Tensor& a, const Tensor& b);
    CHAOS_API Tensor permute(const Tensor& a, const std::vector<uint32>& orders);
    CHAOS_API Tensor sum(const Tensor& a);
    CHAOS_API Tensor sum(const Tensor& a, const std::vector<uint32>& vecdim);
    CHAOS_API Tensor transpose(const Tensor& a);
}