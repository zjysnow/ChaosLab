#include "core/log.hpp"

#include <mutex>
#include <chrono>
#include <iomanip>
#include <iostream>

#ifdef _WIN32
auto time_now = []() { return std::chrono::current_zone()->to_local(std::chrono::system_clock::now()); };
#else
auto time_now = []() {
	time_t time_stamp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	tm time;
	localtime_r(&time_stamp, &time);
	return time;
};
std::ostream& operator<<(std::ostream& stream, const tm& time)
{
	// the result is different from windows
	return stream << time.tm_year + 1990 << "-" << time.tm_mon + 1 << "-" << time.tm_mday 
		<< " " << time.tm_hour << ":" << time.tm_min << ":" << time.tm_sec;
}
#endif

namespace chaos
{
	constexpr const char* const  LogSeverityNames[] = {"INFO", "WARNING", "ERROR", "FATAL"};

	LogMessage::LogMessage(const char* file, int line, const LogSeverity& severity) : severity(severity)
	{
		// remove the path
		auto name = [file]() {
			std::string fname = file;
			for (auto& c : fname) if (c == '\\') c = '/'; // replace slash
			size_t pos = fname.find_last_of('/') + 1;
			return fname.substr(pos);
		};
		message_data << "[" << LogSeverityNames[severity] << " " << time_now() << " " << name() << ":" << line << "] ";
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