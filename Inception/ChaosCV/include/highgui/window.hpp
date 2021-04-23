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
		//virtual void CreatePipeline() = 0;
		virtual void Show() = 0;

		

		//Ptr<VulkanPainter> painter;

	protected:
		// create by VulkanPainter
		static Ptr<VulkanWindow> Create(const std::wstring& name, uint32 width, uint32 height, const VulkanDevice* vkdev = nullptr);

		VulkanPainter* painter;

		virtual void CreateFrameBuffer(const VkRenderPass& render_pass) = 0;
		friend class VulkanPainterImpl;
		uint32 image_count;
		VkSurfaceFormatKHR surface_format;
		VkExtent2D extent;
		std::vector<VkImageView> image_views;
		std::vector<VkFramebuffer> frame_buffers;
	};
}