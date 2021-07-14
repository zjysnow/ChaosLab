#pragma once

#include "core/core.hpp"
#include "core/gpu.hpp"

namespace chaos
{
	class VulkanDevice;
	class Painter;
	class CHAOS_API Window
	{
	public:
		virtual ~Window() = default;
		virtual void Show() = 0;

	protected:
		friend class VulkanPainter;
		static Ptr<Window> Create(const std::wstring& name, uint32 width, uint32 height, const VulkanDevice* vkdev);

		virtual VkFramebuffer* CreateFrameBuffer(const VkRenderPass& render_pass) = 0;

		virtual int height() const = 0;
		virtual int width() const = 0;
		virtual int format() const = 0;

		Painter* painter;
		uint32 image_count;
		

	};
}