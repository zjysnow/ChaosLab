#include "core/file.hpp"
#include "core/log.hpp"

#include <regex>
#include <iostream>
#include <algorithm>


namespace chaos
{
	File::File(const char* data) : buff(data)
	{
		for (auto& c : buff) if (c == '\\') c = '/';

		spos = buff.find_last_of('/') + 1; // last slash pos
		auto file_ = std::string_view(buff.data() + spos); // without path
		ppos = std::min(file_.find_last_of('.'), file_.size());

		auto name_ = 0 == ppos ? file_ : file_.substr(0, ppos);
		auto valid = std::regex_match(std::string(name_), std::regex("[^\\|\\\\/:\\*\\?\"<>]+"));

		LOG_IF(ERROR, valid) << "file name can not contain |\\/:*?\"<>";
	}
	File::File(const std::string& buff) : buff(buff.data()) {}

	File::File(const File& file) : File(file.buff) {}
	File& File::operator=(const File& file)
	{
		return *this = File(file);
	}

	File::File(File&& file) noexcept : buff(std::exchange(file.buff, "")), ppos(std::exchange(file.ppos, 0)), spos(std::exchange(file.spos, 0)) {};
	File& File::operator=(File&& file) noexcept
	{
		std::swap(buff, file.buff);
		std::swap(ppos, file.ppos);
		std::swap(spos, file.spos);
		return *this;
	}
}