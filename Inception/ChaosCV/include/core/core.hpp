#pragma once

#include "core/def.hpp"
#include "core/log.hpp"
#include "core/file.hpp"
#include "core/types.hpp"
#include "core/array.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <functional>

namespace chaos
{
	CHAOS_API std::vector<std::string> Split(const std::string& data, const std::string& delimiter);
}