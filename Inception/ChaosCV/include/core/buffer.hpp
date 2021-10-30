#pragma once

#include "def.hpp"
#include "log.hpp"

namespace chaos
{
	class CHAOS_API Buffer
	{
	public:
		Buffer() = default;

		void* data = nullptr;
		int ref_cnt = 0;
	};
}