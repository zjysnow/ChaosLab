#pragma once

#include "core/def.hpp"

#include <string>

namespace chaos
{
	class CHAOS_API File
	{
	public:
		File() = default;
		File(const char* data);
		File(const std::string & buff);

		File(const File& file);
		File& operator=(const File& file);

		File(File&& file) noexcept;
		File& operator=(File&& file) noexcept;

		const std::string_view path() const noexcept
		{
			return std::string_view(buff.data(), spos);
		}
		const std::string_view name() const noexcept
		{
			return 0 == ppos ? std::string_view(buff.data() + spos) : std::string_view(buff.data() + spos, ppos);
		}
		const std::string_view type() const noexcept
		{
			return 0 == ppos ? std::string_view() : std::string_view(buff.data() + spos + ppos);
		}

		const char* data() const noexcept { return buff.data(); }

	private:
		std::string buff;
		size_t ppos = 0; // last point pose
		size_t spos = 0; // last slash pose
	};
}