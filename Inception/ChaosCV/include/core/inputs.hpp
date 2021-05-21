#pragma once

#include "core/core.hpp"

namespace chaos
{
	class CHAOS_API Inputs
	{
	public:
		Inputs();

	protected:
		void Create(int type, void* obj);

		void* obj;
		int type;
	};
}