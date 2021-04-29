#pragma once

#include "core/core.hpp"

#include "highgui/painter.hpp"

namespace chaos
{
	class VulkanPainter;
	class VulkanDevice;
	class CHAOS_API VulkanWindow
	{
	public:
		virtual ~VulkanWindow() = default;
		virtual void Show() = 0;

	protected:
		friend class VulkanPainterImpl;

		static Ptr<VulkanWindow> Create(const std::wstring& name, uint32 width, uint32 height, const VulkanDevice* vkdev);

		virtual void* CreateFrameBuffer(const VkRenderPass& render_pass) = 0;
		virtual uint32 height() const noexcept = 0;
		virtual uint32 width() const noexcept = 0;
		virtual int image_format() const noexcept = 0;

		VulkanPainter* painter;
		uint32 image_count;
	};
}