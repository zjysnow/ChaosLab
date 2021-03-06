#pragma once

#include "core/core.hpp"
#include "core/types.hpp"
#include "core/tensor.hpp"

namespace chaos
{
	static inline Complex operator+(const Complex& lhs, const Complex& rhs) { return Complex(lhs.re + rhs.re, lhs.im + rhs.im); }
	static inline Complex operator-(const Complex& lhs, const Complex& rhs) { return Complex(lhs.re - rhs.re, lhs.im - rhs.im); }
	static inline Complex operator*(const Complex& lhs, const Complex& rhs) { return Complex(lhs.re * rhs.re - lhs.im * rhs.im, lhs.re * rhs.im + lhs.im * rhs.re); }
	static inline Complex operator/(const Complex& lhs, const Complex& rhs)
	{
		float base = rhs.re * rhs.re + rhs.im * rhs.im;
		float re = (lhs.re * rhs.re + lhs.im * rhs.im) / base;
		float im = (lhs.im * rhs.re - lhs.re * rhs.im) / base;
		return Complex(re, im);
	}

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

	CHAOS_API Tensor diag(const Tensor& a, int d = 0);
	CHAOS_API Tensor div(const Tensor& a, const Tensor& b);
	CHAOS_API Tensor max(const Tensor& a, bool all = false, const Array<int>& vecdim = { 0 });
	CHAOS_API Tensor max(const Tensor& a, const Tensor& b);
	CHAOS_API Tensor mean(const Tensor& a, bool all = false, const Array<int>& vecdim = { 0 });
	CHAOS_API Tensor min(const Tensor& a, bool all = false, const Array<int>& vecdim = {0});
	CHAOS_API Tensor min(const Tensor& a, const Tensor& b);
	CHAOS_API Tensor mul(const Tensor& a, const Tensor& b);
	CHAOS_API Tensor permute(const Tensor& a, const Array<int> orders);
	CHAOS_API Tensor sum(const Tensor& a, bool all = false, const Array<int>& vecdim = { 0 });
	CHAOS_API std::tuple<Tensor, Tensor, Tensor> svd(const Tensor& a);
	CHAOS_API Tensor transpose(const Tensor& a);
	CHAOS_API Tensor var(const Tensor& a, bool all = false, const Array<int>& vecdim = { 0 }, bool unbias = true);

}