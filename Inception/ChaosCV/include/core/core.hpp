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
}