#pragma once

#include "def.hpp"
#include "log.hpp"

namespace chaos
{
	class Buffer
	{
	public:
		size_t capacity = 0;
		void* data = nullptr;
		int ref_cnt = 0;
	};
}