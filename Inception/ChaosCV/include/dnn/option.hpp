#pragma once

#include "core/core.hpp"

namespace chaos
{
	class CHAOS_API Option
	{
	public:
		Option() = default;


		Allocator* blob_allocator = nullptr;
		Allocator* workspace_allocator = nullptr;
	};
}