#include "core/file.hpp"
#include "core/log.hpp"
#include "core/types.hpp"

#include <regex>

namespace chaos
{
	File::File(const char* file) : buf(file)
	{
		if (buf.empty()) return;

		for (auto& b : buf)
		{
			if (b == '/') b = '\\';
		}

		spos = buf.find_last_of('\\') + 1; // last slash pos
		auto remain = std::string_view(buf.data() + spos); // without path
		ppos = remain.find_last_of('.');

		std::string_view name_ = 0 == ppos ? remain : remain.substr(0, ppos);
		auto valid = std::regex_match(std::string(name_), std::regex("[^\\|\\\\/:\\*\\?\"<>]+"));
		CHECK(valid) << "file name can not contain |\\/:*?\"<>";
	}
	File::File(const std::string& file) : File(file.data()) {}

	inline const std::string_view File::path() const noexcept
	{
		return std::string_view(buf.data(), spos);
	}

	inline const std::string_view File::name() const noexcept
	{
		return 0 == ppos ? std::string_view(buf.data() + spos) : std::string_view(buf.data() + spos, ppos);
	}

	inline const std::string_view File::type() const noexcept
	{
		return 0 == ppos ? std::string_view() : std::string_view(buf.data() + spos + ppos);
	}

	inline const char* File::data() const noexcept
	{
		return buf.data();
	}

	std::ostream& operator<<(std::ostream& stream, const File& file)
	{
		return stream << file.buf;
	}
}