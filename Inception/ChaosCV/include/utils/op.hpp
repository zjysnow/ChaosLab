#pragma once

#include "core/core.hpp"
#include "core/tensor.hpp"

namespace chaos
{
	CHAOS_API Tensor operator+(const Tensor& a, const Tensor& b);
	CHAOS_API Tensor operator+(float a, const Tensor& b);
	CHAOS_API Tensor operator+(const Tensor& a, float b);

	CHAOS_API Tensor operator-(const Tensor& a, const Tensor& b);
	CHAOS_API Tensor operator-(float a, const Tensor& b);
	CHAOS_API Tensor operator-(const Tensor& a, float b);

	CHAOS_API Tensor operator*(float a, const Tensor& b);
	CHAOS_API Tensor operator*(const Tensor& a, float b);

	CHAOS_API Tensor operator/(float a, const Tensor& b);
	CHAOS_API Tensor operator/(const Tensor& a, float b);

	void add(const Tensor& a, const Tensor& b, Tensor& c);
	void div(const Tensor& a, const Tensor& b, Tensor& c);
	void mul(const Tensor& a, const Tensor& b, Tensor& c);
	void sub(const Tensor& a, const Tensor& b, Tensor& c);
}