#pragma once

#include "core/def.hpp"

#include <sstream>

namespace chaos
{
	enum // class LogSeverity
	{
		INFO,
		WARNING,
		ERROR,
		FATAL,
	};
	using LogSeverity = int; // to eliminate the warnings for enum

	class CHAOS_API LogMessage
	{
	public:
		LogMessage(const char* file, int line, const LogSeverity& severity);
		~LogMessage();

		std::ostream& stream() { return message_data; };
	private:
		void Flush();

		std::stringstream message_data;
		LogSeverity severity;
	};

	class CHAOS_API LogMessageVoidify
	{
	public:
		LogMessageVoidify() = default;

		// This has to be an operator with a precedence lower than << but
		// higher than ?:
		void operator&(std::ostream&) {}
	};
}