#include "core/core.hpp"

#include <regex>

namespace chaos
{
	std::vector<std::string> Split(const std::string& data, const std::string& delimiter)
	{
		std::regex regex{ delimiter };
		return std::vector<std::string> {
			std::sregex_token_iterator(data.begin(), data.end(), regex, -1),
				std::sregex_token_iterator()
		};
	}

	std::ostream& operator<<(std::ostream& stream, const std::vector<std::string>& list)
	{
		for (size_t i = 0; i < list.size(); i++)
		{
			stream << std::format(",{}"+!i, list[i]);
		}
		return stream;
	}
}