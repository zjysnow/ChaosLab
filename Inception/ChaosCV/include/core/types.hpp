#pragma once

#include "core/def.hpp"
#include <ostream>

namespace chaos
{
	using uchar = unsigned char;
	using int8 = __int8;
	using uint8 = unsigned __int8;
	using int16 = __int16;
	using uint16 = unsigned __int16;
	using int32 = __int32;
	using uint32 = unsigned __int32;
	using int64 = __int64;
	using uint64 = unsigned __int64;
	
	using Flag = int32;

	template<class Type>
	using Ptr = std::shared_ptr<Type>;

	

	enum class LogSeverity
	{
		INFO,
		WARNING,
		ERROR,
		FATAL,
	};

	enum class Depth
	{
		D1 = 1,
		D2 = 2,
		D4 = 4,
		D8 = 8,
	};

	enum class Packing
	{
		CHW = 1,
		C2HW2 = 2,
		C3HW3 = 3,
		C4HW4 = 4,
		C8HW8 = 8,
	};

	template<class Type>
	Type operator*(const Type& val, const Depth& depth)
	{
		return static_cast<Type>(depth) * val;
	}

	template<class Type>
	Type operator*(const Type& val, const Packing& packing)
	{
		return static_cast<Type>(packing) * val;
	}


	class CHAOS_API Point
	{
	public:
		float x;
		float y;
	};

	class CHAOS_API Color
	{
	public:
		float r;
		float g;
		float b;
	};

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

