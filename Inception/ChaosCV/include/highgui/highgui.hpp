#pragma once

#include "core/core.hpp"
#include "core/tensor.hpp"

namespace chaos
{
	class CHAOS_API VulkanWindow
	{
	public:
		virtual ~VulkanWindow() {}

		virtual void Show() = 0;
		void Draw(const std::vector<Point>& pts, const std::vector<Color>& colors, const std::vector<uint16>& indices);

		static Ptr<VulkanWindow> Create(const std::wstring& name, int device_id = GetDefaultGPUIndex(), uint32 width = 800, uint32 height = 600);
	};
}