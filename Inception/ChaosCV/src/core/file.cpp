#include "core/core.hpp"

#include <regex>

namespace chaos
{
	File::File(const wchar_t* file) : buf(file)
	{
		if (buf.empty()) return;

		for (auto& b : buf)
		{
			if (b == '/') b = '\\';
		}

		spos = buf.find_last_of('\\') + 1; // last slash pos
		auto remain = std::wstring_view(buf.data() + spos); // without path
		ppos = remain.find_last_of('.');

		std::wstring_view name_ = 0 == ppos ? remain : remain.substr(0, ppos);
		auto valid = std::regex_match(std::wstring(name_), std::wregex(L"[^\\|\\\\/:\\*\\?\"<>]+"));
		CHECK(valid) << "file name can not contain |\\/:*?\"<>";
	}

	File::File(const std::wstring& file) : File(file.data()) {}

	inline const std::wstring_view File::path() const noexcept
	{
		return std::wstring_view(buf.data(), spos);
	}

	inline const std::wstring_view File::name() const noexcept
	{
		return 0 == ppos ? std::wstring_view(buf.data() + spos) : std::wstring_view(buf.data() + spos, ppos);
	}

	inline const std::wstring_view File::type() const noexcept
	{
		return 0 == ppos ? std::wstring_view() : std::wstring_view(buf.data() + spos + ppos);
	}


	std::wostream& operator<<(std::wostream& wstream, const File& file)
	{
		return wstream << file.buf;
	}

	std::ostream& operator<<(std::ostream& stream, const File& file)
	{
		return stream << (std::string)file;
	}
}

#ifdef _WIN32
#include <io.h>
#include <Windows.h>

chaos::File::operator std::string() const
{
	std::string file;
	int len = WideCharToMultiByte(CP_ACP, 0, buf.data(), -1, NULL, 0, NULL, NULL);
	file.resize(len);

	WideCharToMultiByte(CP_ACP, 0, buf.data(), -1, file.data(), len, NULL, NULL);

	return file;
}


void chaos::GetFileList(const std::wstring& folder, chaos::FileList& list, const std::wstring& types)
{
	CHECK_EQ(0, _waccess(folder.data(), 6)) << "can not access to \"" << folder << "\"";

	HANDLE handle;
	WIN32_FIND_DATA find_data;

	std::wstring root = folder;
	if (root.back() != '\\' || root.back() != '/') root.append(L"\\");

	static std::vector<std::wstring> type_list = chaos::Split(types, L"\\|");

	handle = FindFirstFile((root + L"*.*").data(), &find_data);
	if (handle != INVALID_HANDLE_VALUE)
	{
		do
		{
			if ('.' == find_data.cFileName[0])
			{
				continue;
			}
			else if (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			{
				GetFileList(root + find_data.cFileName, list, types);
			}
			else
			{
				std::wstring file_name = find_data.cFileName;

				size_t pos = file_name.find_last_of('.') + 1;
				std::wstring type = file_name.substr(pos);
				if (L"*" == types || std::find(type_list.begin(), type_list.end(), type) != type_list.end())
				{
					list.push_back(File(root + file_name));
				}
			}
		} while (FindNextFile(handle, &find_data));
	}

	FindClose(handle);
}
#else
void chaos::GetFileList(const std::wstring& folder, chaos::FileList& list, const std::wstring& types = L"*")
{
	// do something here
}
#endif