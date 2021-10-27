#pragma once

#include "core/def.hpp"

#include <ostream>
#include <format>

namespace chaos
{
	class CHAOS_API Complex
	{
	public:
		constexpr Complex() {}
		constexpr Complex(float re, float im = 0) : re(re), im(im) {}

		Complex conj() const noexcept { return Complex(re, -im); }

		float re = 0.f; // real
		float im = 0.f; // image
	};

	static inline constexpr Complex operator ""i(long double im) { return Complex(0, static_cast<float>(im)); }
	static inline constexpr Complex operator ""i(unsigned long long im) { return Complex(0, static_cast<float>(im)); }

	static inline std::ostream& operator<<(std::ostream& stream, const Complex& complex)
	{
		return stream << std::format("{0:}{1:+}i", complex.re, complex.im);
	}
	template<class CharT>
	struct std::formatter<chaos::Complex, CharT> : std::formatter<std::string, CharT>
	{
		template<class FormatContext>
		auto format(const chaos::Complex& complex, FormatContext& fc)
		{
			std::string val = std::format("{}{:+}i", complex.re, complex.im);
			return std::formatter<std::string, CharT>::format(val, fc);
		}
	};
}