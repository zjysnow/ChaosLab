#pragma once

#include "core/def.hpp"

#include <string>
#include <vector>
#include <format>

namespace chaos
{
	class CHAOS_API File
	{
	public:
		File() = default;
		File(const char* file);
		File(const std::string& file);

		const std::string_view path() const noexcept;
		const std::string_view name() const noexcept;
		const std::string_view type() const noexcept;

		const char* data() const noexcept;
		CHAOS_API friend std::ostream& operator<<(std::ostream& stream, const File& file);
	private:
		std::string buf;
		size_t ppos = -1; // last point pose
		size_t spos = -1; // last slash pose
	};

	using FileList = std::vector<File>;
	CHAOS_API void GetFileList(const std::string& folder, FileList& list, const std::string& = "*");

	template<class CharT>
	struct std::formatter<chaos::File, CharT> : std::formatter<const char*, CharT>
	{
		template<class FormatContext>
		auto format(const chaos::File& file, FormatContext& fc)
		{
			return std::formatter<const char*, CharT>::format(file.data(), fc);
		}
	};
}