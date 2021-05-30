#include "core/core.hpp"

namespace chaos
{
	std::ostream& operator<<(std::ostream& stream, const std::vector<std::string>& list)
	{
		for (size_t i = 0; i < list.size(); i++)
		{
			stream << std::format(",{}"+!i, list[i]);
		}
		return stream;
	}
}