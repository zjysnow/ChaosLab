#pragma once

#include "core/def.hpp"
#include "core/log.hpp"
#include "core/file.hpp"
#include "core/types.hpp"
#include "core/array.hpp"

#include <format>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

namespace chaos
{
	CHAOS_API std::ostream& operator<<(std::ostream& stream, const std::vector<std::string>& list);

	constexpr uint64 prime = 0x100000001B3ULL;
	constexpr uint64 basis = 0xCBF29CE484222325ULL;
	static constexpr uint64 Hash(const char* data, uint64 last_value = basis)
	{
		return *data ? Hash(data + 1, (*data ^ last_value) * prime) : last_value;
	}
	inline static constexpr uint64 Hash(const std::string& str)
	{
		return Hash(str.data());
	}
	inline static constexpr uint64 operator ""_hash(const char* data, size_t)
	{
		return Hash(data);
	}
	inline static constexpr uint64 operator ""h(const char* data, size_t)
	{
		return Hash(data);
	}
}