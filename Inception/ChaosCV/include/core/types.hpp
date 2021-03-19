#pragma once

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
}