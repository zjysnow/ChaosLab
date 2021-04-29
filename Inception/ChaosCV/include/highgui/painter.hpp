#pragma once

#include "core/core.hpp"

#include "highgui/highgui.hpp"

namespace chaos
{
	class GraphicsCommand;
	class GraphicsPipeline;
	class VulkanWindow;
	class CHAOS_API VulkanPainter
	{
	public:
		friend class VulkanWindowImpl;

		virtual Ptr<VulkanWindow> CreateWindow(const std::wstring& name, uint32 width, uint32 height) = 0;
		virtual void Draw(const std::vector<Point>& pts, const std::vector<Color>& colors, const std::vector<uint16>& ind) = 0;

		virtual ~VulkanPainter();

		static Ptr<VulkanPainter> Create(const File& vert, const File& frag, uint32 device_index = GetDefaultGPUIndex());

		int front_face = FRONT_FACE_CLOCKWISE; // see enum FrontFace
		int polygon_mode = POLYGON_MODE_LINE; // see enum PolygonMode
		int topoloty = PRIMITIVE_TOPOLOGY_LINE_LIST; // see enum PrimitiveTopology

	protected:
		VulkanPainter(const VulkanDevice* vkdev);

		Ptr<VulkanWindow> window;

		GraphicsPipeline* pipeline;
		GraphicsCommand* command;

		VulkanAllocator* allocator;
		VulkanAllocator* staging_allocator;

		const VulkanDevice* vkdev;
	};
}