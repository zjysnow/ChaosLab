#pragma once

#include "core/core.hpp"
#include "dnn/layer.hpp"

namespace chaos
{
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

    CHAOS_API void add(const Tensor& a, const Tensor& b, Tensor& c);
    CHAOS_API Tensor cross(const Tensor& a, const Tensor& b);
    CHAOS_API void div(const Tensor& a, const Tensor& b, Tensor& c);
    CHAOS_API Tensor dot(const Tensor& a, const Tensor& b);
    CHAOS_API void gemm(bool transA, bool transB, const Tensor& a, const Tensor& b, float alpha, bool transC, Tensor& c, float beta);
    CHAOS_API Tensor mean(const Tensor& a);
    CHAOS_API Tensor mean(const Tensor& a, const std::vector<uint32>& vecdim = { 0 });
    CHAOS_API Tensor mul(const Tensor& a, const Tensor& b);
    CHAOS_API void mul(const Tensor& a, const Tensor& b, Tensor& c);
    CHAOS_API Tensor norm(const Tensor& a, float p = 2.f);
    CHAOS_API Tensor normalize(const Tensor& a, const std::string& type = "norm");
    CHAOS_API Tensor permute(const Tensor& a, const std::vector<uint32>& orders);
    CHAOS_API void sub(const Tensor& a, const Tensor& b, Tensor& c);
    CHAOS_API Tensor sum(const Tensor& a);
    CHAOS_API Tensor sum(const Tensor& a, const std::vector<uint32>& vecdim = { 0 });
    CHAOS_API Tensor transpose(const Tensor& a);
    CHAOS_API void transpose(const Tensor& a, Tensor& p);

    //CHAOS_API std::tuple<Tensor, Tensor, Tensor> svd(const Tensor& A);
}