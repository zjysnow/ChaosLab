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

	CHAOS_API void add(const Tensor& a, const Tensor& b, Tensor& c);
	CHAOS_API void div(const Tensor& a, const Tensor& b, Tensor& c);
	CHAOS_API void mul(const Tensor& a, const Tensor& b, Tensor& c);
	CHAOS_API void permute(const Tensor& a, const Array<uint32>& orders, Tensor& b);
	CHAOS_API void sub(const Tensor& a, const Tensor& b, Tensor& c);
	CHAOS_API void svd(const Tensor& a, Tensor& w); // NO_UV
	CHAOS_API void svd(const Tensor& a, Tensor& w, Tensor& u, Tensor& vt, bool full_uv = false);
	CHAOS_API void transpose(const Tensor& a, Tensor& b);
	
	
	
	void invert(const Tensor& a, Tensor& b);
}