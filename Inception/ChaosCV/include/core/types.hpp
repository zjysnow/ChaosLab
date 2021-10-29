#pragma once

#include "core/def.hpp"

#include <iostream>
#include <cmath>

namespace chaos
{
	class CHAOS_API Complex
	{
	public:
		constexpr Complex() {}
		constexpr Complex(float re, float im = 0) : re(re), im(im) {}

		Complex conj() const noexcept { return Complex(re, -im); }
		float abs() const noexcept { return sqrtf(re * re + im * im); }
		float angle() const noexcept { return atan2f(im, re); }

		float re = 0.f; // real
		float im = 0.f; // image
	};
	static inline constexpr Complex operator ""i(long double im) { return Complex(0, static_cast<float>(im)); }
	static inline constexpr Complex operator ""i(unsigned long long im) { return Complex(0, static_cast<float>(im)); }

	static inline Complex operator+(const Complex& lhs, const Complex& rhs) { return Complex(lhs.re + rhs.re, lhs.im + rhs.im); }
	static inline Complex operator-(const Complex& lhs, const Complex& rhs) { return Complex(lhs.re - rhs.re, lhs.im - rhs.im); }

	static inline std::ostream& operator<<(std::ostream& stream, const Complex& complex)
	{
		return stream << complex.re << std::showpos << complex.im << "i" << std::noshowpos;
	}
}