#include "core/core.hpp"
#include "core/buffer.hpp"
#include "core/allocator.hpp"

#include <regex>
#include <Windows.h>

namespace chaos
{
	std::vector<std::string> Split(const std::string& data, const std::string& delimiter)
	{
		std::regex regex{ delimiter };
		return std::vector<std::string> {
			std::sregex_token_iterator(data.begin(), data.end(), regex, -1),
				std::sregex_token_iterator()
		};
	}

	//std::string UnicodeToUTF8(const std::wstring& unicode)
	//{
	//	int len = WideCharToMultiByte(CP_UTF8, 0, unicode.data(), -1, NULL, 0, NULL, NULL);
	//	std::string utf8;
	//	utf8.resize(len+1);
	//	WideCharToMultiByte(CP_UTF8, 0, unicode.data(), -1, utf8.data(), len, NULL, NULL);
	//	return utf8;

	//}
	//std::wstring UTF8ToUnicode(const std::string& utf8)
	//{
	//	return std::wstring();
	//}

	int chaos_vsnprintf(char* buf, int len, const char* fmt, va_list args)
	{
		if (len <= 0) return len == 0 ? 1024 : -1;
		int res = _vsnprintf_s(buf, len, _TRUNCATE, fmt, args);
		// ensure null terminating on VS
		if (res >= 0 && res < len)
		{
			buf[res] = 0;
			return res;
		}
		else
		{
			buf[len - 1] = 0; // truncate happened
			return res >= len ? res : (len * 2);
		}
	}

	std::string Format(const char* fmt, ...)
	{
		AutoBuffer<char, 1024> buf;

		for (; ; )
		{
			va_list va;
			va_start(va, fmt);
			int bsize = static_cast<int>(buf.size());
			int len = chaos_vsnprintf(buf.data(), bsize, fmt, va);
			va_end(va);

			//CHECK(len >= 0) << "check format string for errors";
			if (len >= bsize)
			{
				buf.Resize(len + 1LL);
				continue;
			}
			buf[bsize - 1LL] = 0;
			return std::string(buf.data(), len);
		}
	}

	std::ostream& operator<<(std::ostream& stream, const std::vector<std::string>& list)
	{
		for (size_t i = 0; i < list.size(); i++)
		{
			stream << Format(",%d" + !i, list[i]);
		}
		return stream;
	}
}