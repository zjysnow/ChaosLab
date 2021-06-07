#pragma once

#include "core/core.hpp"
#include "core/allocator.hpp"

namespace chaos
{
	class CHAOS_API Option
	{
	public:
		Option() = default;

		// enable vulkan compute
		bool use_vulkan_compute = false;

		Allocator* blob_allocator = nullptr;

		VulkanAllocator* blob_vkallocator = nullptr;
		VulkanAllocator* staging_vkallocator = nullptr;
	};
}