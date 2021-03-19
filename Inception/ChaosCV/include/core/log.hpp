#pragma once

#include "core/def.hpp"
#include "core/types.hpp"

#include <sstream>

namespace chaos
{
	class CHAOS_API LogMessage
	{
	public:
		LogMessage(const wchar_t* file, int line, const LogSeverity& severity);
		~LogMessage();

		std::wostream& wstream();
	private:
		void Flush();

		std::wstringstream message_data;
		LogSeverity severity;
	};

	class CHAOS_API LogMessageVoidify
	{
	public:
		LogMessageVoidify() = default;
		
		// This has to be an operator with a precedence lower than << but
		// higher than ?:
		void operator&(std::wostream&) {}
	};
}