#include "core/core.hpp"

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


	std::ostream& operator<<(std::ostream& wstream, const File& file)
	{
		return wstream << file.buf;
	}

	//std::ostream& operator<<(std::ostream& stream, const File& file)
	//{
	//	return stream << (std::string)file;
	//}
}

#ifdef _WIN32
#include <io.h>
#include <Windows.h>

//chaos::File::operator std::string() const
//{
//	std::string file;
//	int len = WideCharToMultiByte(CP_ACP, 0, buf.data(), -1, NULL, 0, NULL, NULL);
//	file.resize(len);
//
//	WideCharToMultiByte(CP_ACP, 0, buf.data(), -1, file.data(), len, NULL, NULL);
//
//	return file;
//}


void chaos::GetFileList(const std::string& folder, chaos::FileList& list, const std::string& types)
{
	CHECK_EQ(0, _access(folder.data(), 6)) << "can not access to \"" << folder << "\"";

	HANDLE handle;
	WIN32_FIND_DATAA find_data;

	std::string root = folder;
	if (root.back() != '\\' || root.back() != '/') root.append("\\");

	static std::vector<std::string> type_list = chaos::Split(types, "\\|");

	handle = FindFirstFileA((root + "*.*").data(), &find_data);
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
				std::string file_name = find_data.cFileName;

				size_t pos = file_name.find_last_of('.') + 1;
				std::string type = file_name.substr(pos);
				if ("*" == types || std::find(type_list.begin(), type_list.end(), type) != type_list.end())
				{
					list.push_back(File(root + file_name));
				}
			}
		} while (FindNextFileA(handle, &find_data));
	}

	FindClose(handle);
}
#else
void chaos::GetFileList(const std::wstring& folder, chaos::FileList& list, const std::wstring& types = L"*")
{
	// do something here
}
#endif