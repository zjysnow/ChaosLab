#pragma once

#include "core/core.hpp"
#include "core/allocator.hpp"

namespace chaos
{
	class Option
	{
	public:
		Allocator* blob_allocator = nullptr;
		Allocator* workspace_allocator = nullptr;
	};
}