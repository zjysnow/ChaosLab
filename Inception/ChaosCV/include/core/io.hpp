#pragma once

#include "core/types.hpp"
#include "core/array.hpp"

#include <iostream>

namespace chaos
{
	static inline std::ostream& operator<<(std::ostream& stream, const Complex& complex)
	{
		return stream << complex.re << (complex.im < 0 ? "" : "+") << complex.im << "i";
	}
	//static inline std::ostream& operator<<(std::ostream& stream, const Complex& complex)
	//{
	//	return stream << std::format("{0:}{1:+}i", complex.re, complex.im);
	//}
	//template<class CharT>
	//struct std::formatter<chaos::Complex, CharT> : std::formatter<std::string, CharT>
	//{
	//	template<class FormatContext>
	//	auto format(const chaos::Complex& complex, FormatContext& fc)
	//	{
	//		std::string val = std::format("{}{:+}i", complex.re, complex.im);
	//		return std::formatter<std::string, CharT>::format(val, fc);
	//	}
	//};
}