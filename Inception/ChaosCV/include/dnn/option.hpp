#pragma once

#include "core/core.hpp"

namespace chaos
{
	class CHAOS_API Option
	{
	public:
		Option() = default;

		// light mode
		// intermediate blob will be recycled when enabled
		// disenabled by default
		bool light_mode = false;

		// enable vulkan compute
		bool use_vulkan_compute = false;

		Allocator* blob_allocator = nullptr;
		Allocator* workspace_allocator = nullptr;
	};
}