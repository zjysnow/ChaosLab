#pragma once

#include "core/core.hpp"
#include "core/allocator.hpp"

namespace chaos
{
	class CHAOS_API Option
	{
	public:
		Option() = default;

		Allocator* allocator = nullptr;

		VulkanAllocator* vkallocator = nullptr;
		VulkanAllocator* staging_vkallocator = nullptr;
	};
}