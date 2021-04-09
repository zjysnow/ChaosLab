#pragma once

#include "core/types.hpp"

#include <sstream>

namespace chaos
{
	class CHAOS_API LogMessage
	{
	public:
		LogMessage(const char* file, int line, const LogSeverity& severity);
		~LogMessage();

		std::ostream& stream();
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