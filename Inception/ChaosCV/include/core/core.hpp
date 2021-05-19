#pragma once

#include "core/def.hpp"
#include "core/log.hpp"
#include "core/file.hpp"
#include "core/types.hpp"
#include "core/allocator.hpp"

namespace chaos
{
	/// <summary>Split the wstring data by delimiter</summary>
	CHAOS_API std::vector<std::string> Split(const std::string& data, const std::string& delimiter);

	/// <summary>
	/// <para>Returns a text wstring formatted using the printf-like expression</para>
	/// <para>The function acts like sprintf but forms and returns an STL wstring. It can be used to form an error</para>
	/// <para>message in the Exception constructor.</para>
	/// </summary>
	/// <param name="fmt">printf-compatible formatting specifiers.</param>
	CHAOS_API std::string Format(const char* fmt, ...);

	constexpr uint64 prime = 0x100000001B3ULL;
	constexpr uint64 basis = 0xCBF29CE484222325ULL;
	constexpr uint64 Hash(const char* data, uint64 last_value = basis)
	{
		return *data ? Hash(data + 1, (*data ^ last_value) * prime) : last_value;
	}
	constexpr uint64 operator ""_hash(const char* data , size_t)
	{
		return Hash(data);
	}
}