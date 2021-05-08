#pragma once

#include "core/core.hpp"
#include "core/tensor.hpp"

#include "dnn/layer.hpp"

namespace chaos
{
	static  inline float radians(float degrees) { return degrees * 0.01745329251994329576923690768489f; }

	class CHAOS_API Operator
	{
	public:
		static Tensor Add(const Tensor& a, const Tensor& b);
		static Tensor Sub(const Tensor& a, const Tensor& b);
		static Tensor Mul(const Tensor& a, const Tensor& b);
		static Tensor Div(const Tensor& a, const Tensor& b);

		static Tensor Mul(const Tensor& a, float b);
		static Tensor Div(const Tensor& a, float b);


		static Tensor GEMM(const Tensor& a, const Tensor& b, float alpha = 1.f, const Tensor& c = Tensor(), float beta = 0.f);

		static float Dot(const Tensor& a, const Tensor& b);
		static Tensor Cross(const Tensor& a, const Tensor& b);

		static Tensor L2Norm(const Tensor& a);

		static void Transpose(const Tensor& a, Tensor& b); // b = a^T;
	private:
		Operator() = delete;

		static Ptr<Layer> binary_op;
		static Ptr<Layer> cross;
		static Ptr<Layer> dot;
		static Ptr<Layer> gemm;
		static Ptr<Layer> normalize;
		static Ptr<Layer> permute;
	};
}