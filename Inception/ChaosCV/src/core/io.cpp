#include "core/io.hpp"

namespace chaos
{
	std::ostream& operator<<(std::ostream& stream, const GPUInfo& info)
	{
		return stream << std::format("[{} {}]", info.device_name, info.type);
	}
}