#pragma once

#include "core/core.hpp"

namespace chaos
{
	class CHAOS_API PipelineCache
	{
	public:
		void NewPipeline(VkShaderModule shader_module);
	};
}