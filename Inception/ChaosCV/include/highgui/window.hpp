#pragma once

#include "core/core.hpp"

#include "highgui/painter.hpp"

namespace chaos
{
	class CHAOS_API VulkanWindow
	{
	public:
		virtual ~VulkanWindow() = default;
		virtual void CreatePipeline() = 0;
		virtual void Show() = 0;

		static Ptr<VulkanWindow> Create(const std::wstring& name, uint32 width, uint32 height, int device_id = GetDefaultGPUIndex());

		Ptr<VulkanPainter> painter;
	};
}