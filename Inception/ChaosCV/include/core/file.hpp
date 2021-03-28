#pragma once

#include "core/def.hpp"

#include <vector>
#include <string_view>

namespace chaos
{
	class CHAOS_API File
	{
	public:
		File() = default;
		File(const wchar_t* file);
		File(const std::wstring& file);

		const std::wstring_view path() const noexcept;
		const std::wstring_view name() const noexcept;
		const std::wstring_view type() const noexcept;

		operator std::wstring() const noexcept { return buf; }
		operator std::string() const;
		CHAOS_API friend std::wostream& operator<<(std::wostream& wstream, const File& file);
		CHAOS_API friend std::ostream& operator<<(std::ostream& stream, const File& file);
	private:
		std::wstring buf;
		size_t ppos = -1; // last point pose
		size_t spos = -1; // last slash pose
	};
	using FileList = std::vector<File>;

	CHAOS_API void GetFileList(const std::wstring& folder, FileList& list, const std::wstring& types = L"*");
}