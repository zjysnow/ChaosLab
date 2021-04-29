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

		std::function<Tensor()> CreateUniformObject = []()->Tensor {
			Tensor ubo = Tensor(Shape(3, 4, 4), DataType::D4, Packing::CHW);
			memset(ubo.data, 0, ubo.total() * sizeof(float));
			for (int i = 0; i < 3; i++)
			{
				ubo[i * 16] = 1;
				ubo[i * 16 + 5] = 1;
				ubo[i * 16 + 10] = 1;
				ubo[i * 16 + 15] = 1;
			}
			return ubo;
		};
	protected:
		VulkanPainter(const VulkanDevice* vkdev);

		virtual void UpdateUniformBuffer(uint32 image_index) = 0;
		Ptr<VulkanWindow> window;

		GraphicsPipeline* pipeline;
		GraphicsCommand* command;

		VulkanAllocator* allocator;
		VulkanAllocator* staging_allocator;

		const VulkanDevice* vkdev;
	};
}