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
		File(const char* file);
		File(const std::string& file);

		const std::string_view path() const noexcept;
		const std::string_view name() const noexcept;
		const std::string_view type() const noexcept;

		operator std::string() const noexcept { return buf; }
		//operator std::string() const;
		//CHAOS_API friend std::wostream& operator<<(std::wostream& wstream, const File& file);
		CHAOS_API friend std::ostream& operator<<(std::ostream& stream, const File& file);
	private:
		std::string buf;
		size_t ppos = -1; // last point pose
		size_t spos = -1; // last slash pose
	};
	using FileList = std::vector<File>;

	CHAOS_API void GetFileList(const std::string& folder, FileList& list, const std::string& types = "*");
}