#include "core/core.hpp"

#include <iostream>
#include <mutex>

namespace chaos
{
	std::ostream& operator<<(std::ostream& wstream, const LogSeverity& severity)
	{
		switch (severity)
		{
		case LogSeverity::INFO:
			return wstream << "INFO";
		case LogSeverity::WARNING:
			return wstream << "WARNING";
		case LogSeverity::ERROR:
			return wstream << "ERROR";
		case LogSeverity::FATAL:
			return wstream << "FATAL";
		default:
			return wstream << "UNK"; // never reachable
		}
	}

	LogMessage::LogMessage(const char* file, int line, const LogSeverity& severity) : severity(severity)
	{
		File file_ = file;

		// Get time stamp
		time_t time_stamp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
		tm time;
		localtime_s(&time, &time_stamp);

		message_data << "[" << severity 
			<< Format(" %04d-%02d-%02d %02d:%02d:%02d ", time.tm_year + 1990, time.tm_mon + 1, time.tm_mday, time.tm_hour, time.tm_min, time.tm_sec) 
			<< file_.name() << "." << file_.type() << ":" << line << "] ";;
	}

	LogMessage::~LogMessage()
	{
		Flush();
		if (severity == LogSeverity::FATAL)
		{
			abort();
		}
	}

	std::ostream& chaos::LogMessage::stream()
	{
		return message_data;
	}

	void LogMessage::Flush()
	{
		static std::mutex mtx;
		std::lock_guard lock(mtx);

		std::string message = message_data.str();
		std::cout << message << std::endl;
	}
}