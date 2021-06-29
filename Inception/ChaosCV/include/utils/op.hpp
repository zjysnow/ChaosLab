#pragma once

#include "core/core.hpp"
#include "core/tensor.hpp"

namespace chaos
{
	// just to make it easier to use

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

	CHAOS_API Tensor sum(const Tensor& a, bool all = false, const Array<int>& vedcim = { 0 });
}