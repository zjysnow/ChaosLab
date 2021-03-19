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
}