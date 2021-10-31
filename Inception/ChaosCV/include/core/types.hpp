#pragma once

#include "core/def.hpp"

#include <iostream>
#include <cmath>

namespace chaos
{
	using uchar = unsigned char;

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
	static inline Complex operator*(const Complex& lhs, const Complex& rhs) 
	{
		// z1=a+bi£¬z2=c+di
		// z1*z2 = (a+bi)(c+di) = (ac-bd)+(bc+ad)i
		float re = lhs.re * rhs.re - lhs.im * rhs.im; // ac - bd
		float im = lhs.im * rhs.re + lhs.re * rhs.im; // bc + ad
		return Complex(re, im);
	}
	static inline Complex operator/(const Complex& lhs, const Complex& rhs) 
	{
		// z1=a+bi£¬z2=c+di
		// z1/z2=z1*conj(z2)/z2*conj(z2)
		float divisor = rhs.re * rhs.re + rhs.im * rhs.im;
		float re = (lhs.re * rhs.re + lhs.im * rhs.im) / divisor; // ac + bd
		float im = (lhs.im * rhs.re - lhs.re * rhs.im) / divisor; // bc - ad
		return Complex(re, im); 
	}

	static inline std::ostream& operator<<(std::ostream& stream, const Complex& complex)
	{
		return stream << complex.re << std::showpos << complex.im << "i" << std::noshowpos;
	}
}