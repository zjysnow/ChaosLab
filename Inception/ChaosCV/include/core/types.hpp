#pragma once

#include "core/def.hpp"

namespace chaos
{
	using int8 = char;
	using uint8 = unsigned char;
	

	enum class LogSeverity
	{
		INFO,
		WARNING,
		ERROR,
		FATAL,
	};

	enum class DataType
	{
		D1 = 1, // int8 or uint8
		D2 = 2, // float16, int16 or uint16
		D4 = 4, // float, int32 or uint32
		D8 = 8, // double, int64 or uint64
	};

	enum class Packing
	{
		CHW = 1, //scalar
		C2HW2 = 2, // complex
		C3HW3 = 3, // 3-channel image
		C4HW4 = 4, // 4-channel image, sse or neon
		C8HW8 = 8, // avx or fp16
	};

}