#pragma once

#include "core/core.hpp"

namespace chaos
{
	class CHAOS_API VulkanWindow
	{
	public:
		virtual ~VulkanWindow() = default;

		virtual void Show() = 0;
		virtual void Draw(const std::vector<Point>& pts, const std::vector<Color>& colors, const std::vector<uint16>& inds) = 0;

		static Ptr<VulkanWindow> Create(const std::wstring& name, int device_id = GetDefaultGPUIndex(), uint32 width = 800, uint32 height = 600);
	};
}