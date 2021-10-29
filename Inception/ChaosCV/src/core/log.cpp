#include "core/log.hpp"
#include "core/file.hpp"

#include <chrono>
#include <mutex>
#include <iostream>

namespace chaos
{
	constexpr const char* const  LogSeverityNames[] = {"INFO", "WARNING", "ERROR", "FATAL"};

	LogMessage::LogMessage(const char* file, int line, const LogSeverity& severity) : severity(severity)
	{
		File code = file;
		auto now = std::chrono::current_zone()->to_local(std::chrono::system_clock::now());
		message_data << "[" << LogSeverityNames[severity] << " " << now << " " << code.name() << code.type() << ":" << line << "] ";
	}

	LogMessage::~LogMessage()
	{
		Flush();
		if (FATAL == severity)
		{
			abort();
		}
	}

	void LogMessage::Flush()
	{
		static std::mutex mtx;
		std::lock_guard lock(mtx);

		std::string message = message_data.str();
		std::cout << message << std::endl;
	}
}