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
		static void Add(const Tensor& a, const Tensor& b, Tensor& c);
		static void Add(float a, const Tensor& b, Tensor& c);

		static void Cross(const Tensor& a, const Tensor& b, Tensor& c);

		static void Diag(const Tensor& a, Tensor& b);
		static void Div(const Tensor& a, const Tensor& b, Tensor& c);
		static void Div(float a, const Tensor& b, Tensor& c);
		static void Dot(const Tensor& a, const Tensor& b, Tensor& c);

		// c = alpha * a * b + beta * c
		static void GEMM(const Tensor& a, const Tensor& b, float alpha, Tensor& c, float beta);

		static void Mean(const Tensor& a, int dim, Tensor& m);
		static void Mul(const Tensor& a, const Tensor& b, Tensor& c);
		static void Mul(float a, const Tensor& b, Tensor& c);

		static void Norm(const Tensor& a, float p, Tensor& n);
		static void Normalize(const Tensor& a, int dim, const std::string& method, float p1, float p2, Tensor& n);

		static void Pow(const Tensor& a, float e, Tensor& c);

		static void Sub(const Tensor& a, const Tensor& b, Tensor& c);
		static void Sub(float a, const Tensor& b, Tensor& c);
		static void Sum(const Tensor& a, int dim, Tensor& b);

		static void Transpose(const Tensor& a, Tensor& b);

	private:
		Operator() = delete;

		static Ptr<Layer> binary_op;
		static Ptr<Layer> cross;
		static Ptr<Layer> diag;
		static Ptr<Layer> dot;
		static Ptr<Layer> gemm;
		static Ptr<Layer> mean;
		static Ptr<Layer> norm;
		static Ptr<Layer> normalize;
		static Ptr<Layer> permute;
		static Ptr<Layer> sum;
		
	};

	CHAOS_API Tensor operator+(const Tensor& lhs, const Tensor& rhs);
	CHAOS_API Tensor operator+(const Tensor& lhs, float rhs);
	CHAOS_API Tensor operator+(float lhs, const Tensor& rhs);

	CHAOS_API Tensor operator-(const Tensor& lhs, const Tensor& rhs);
	CHAOS_API Tensor operator-(const Tensor& lhs, float rhs);
	CHAOS_API Tensor operator-(float lhs, const Tensor& rhs);
	
	CHAOS_API Tensor operator*(const Tensor& lhs, const Tensor& rhs); // GEMM
	CHAOS_API Tensor operator*(const Tensor& lhs, float rhs);
	CHAOS_API Tensor operator*(float lhs, const Tensor& rhs);

	CHAOS_API Tensor operator/(const Tensor& lhs, const Tensor& rhs);
	CHAOS_API Tensor operator/(const Tensor& lhs, float rhs);
	CHAOS_API Tensor operator/(float lhs, const Tensor& rhs);

	// just like Matlab for easy use

	CHAOS_API Tensor cross(const Tensor& a, const Tensor& b);
	CHAOS_API Tensor diag(const Tensor& a, int k = 0);
	CHAOS_API float dot(const Tensor& a, const Tensor& b);
	CHAOS_API Tensor mean(const Tensor& a, int dim = 0);
	CHAOS_API float norm(const Tensor& a, float p = 2.f);
	CHAOS_API Tensor normalize(const Tensor& a, int dim = 0, const std::string& method = "norm", float p1 = 2.f, float p2 = 0.f);
	CHAOS_API Tensor sum(const Tensor& a, int dim = 0);
	
}