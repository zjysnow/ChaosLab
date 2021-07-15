#pragma once

#include "core/core.hpp"
#include "core/tensor.hpp"
#include "core/gpu.hpp"

#include <functional>

namespace chaos
{
	class Window;
	class VulkanDevice;
	class GraphicsPipeline;
	class GraphicsCommand;
	class VulkanAllocator;
	class CHAOS_API Painter
	{
	public:
		virtual ~Painter();
		virtual Ptr<Window> CreateWindow(const std::wstring& name, uint32 width = 800, uint32 height = 600) = 0;
		virtual void Draw(const std::vector<Point>& pts, const std::vector<Color>& colors, const std::vector<uint16>& ind) = 0;

		static Ptr<Painter> Create(const File& vert, const File& frag, uint32 device_index = GetDefaultGPUIndex());
		//static Ptr<Painter> Plot(uint32 device_index = GetDefaultGPUIndex());

		

		std::function<Tensor()> CreateUniformObject = []()->Tensor {
			Tensor ubo = Tensor::zeros(Shape(3, 4, 4));
			memset(ubo.data, 0, ubo.total() * sizeof(float));
			for (size_t i = 0; i < 3; i++)
			{
				ubo[i * 16] = 1;
				ubo[i * 16 + 5] = 1;
				ubo[i * 16 + 10] = 1;
				ubo[i * 16 + 15] = 1;
			}
			return ubo;
		};

		int front_face = FRONT_FACE_CLOCKWISE; // see enum FrontFace
		int polygon_mode = POLYGON_MODE_LINE; // see enum PolygonMode
		int topoloty = PRIMITIVE_TOPOLOGY_LINE_LIST; // see enum PrimitiveTopology
		int cull_mode = CULL_MODE_NONE; // see enum CullModeFlag

	protected:
		friend class VulkanWindow;

		Painter(const VulkanDevice* vkdev);
		virtual void UpdateUniformBuffer(uint32 image_index) = 0;

		Ptr<Window> window;

		GraphicsPipeline* pipeline;
		GraphicsCommand* command;

		VulkanAllocator* allocator;
		VulkanAllocator* staging_allocator;

		const VulkanDevice* vkdev;
	};

}