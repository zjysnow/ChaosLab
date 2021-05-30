#include "core/log.hpp"
#include "core/file.hpp"
#include "core/types.hpp"

#include <map>
#include <mutex>
#include <chrono>
#include <iostream>

namespace chaos
{
	template<class CharT>
	struct std::formatter<chaos::LogSeverity, CharT> : std::formatter<const char* , CharT >
	{
		template<class FormatContext>
		auto format(const chaos::LogSeverity& severity, FormatContext& fc)
		{
			switch (severity)
			{
			case chaos::LogSeverity::INFO:
				return std::formatter<const char*, CharT>::format("INFO", fc);
			case chaos::LogSeverity::WARNING:
				return std::formatter<const char*, CharT>::format("WARNING", fc);
			case chaos::LogSeverity::ERROR:
				return std::formatter<const char*, CharT>::format("ERROR", fc);
			case chaos::LogSeverity::FATAL:
				return std::formatter<const char*, CharT>::format("FATAL", fc);
			default:
				return std::formatter<const char*, CharT>::format("INVALID", fc); // never reachable
			}
		}
	};

	LogMessage::LogMessage(const char* file, int line, const LogSeverity& severity) : severity(severity)
	{
		File file_ = file;
		auto now = std::chrono::current_zone()->to_local(std::chrono::system_clock::now());
		message_data << std::format("[{} {} {}{}:{}] ", severity, now, file_.name(), file_.type(), line);
	}
	LogMessage::~LogMessage()
	{
		Flush();
		if (LogSeverity::FATAL == severity)
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