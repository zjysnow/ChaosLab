#pragma once

#include "core/core.hpp"

namespace chaos
{
	// create pipeline
	// setting
	// cmd.RecordPipeline
	enum FrontFace
	{
		FRONT_FACE_COUNTER_CLOCKWISE = 0,
		FRONT_FACE_CLOCKWISE = 1,
		FRONT_FACE_MAX_ENUM = 0x7FFFFFFF
	};

	enum PolygonMode
	{
		POLYGON_MODE_FILL = 0,
		POLYGON_MODE_LINE = 1,
		POLYGON_MODE_POINT = 2,
		POLYGON_MODE_FILL_RECTANGLE_NV = 1000153000,
		POLYGON_MODE_MAX_ENUM = 0x7FFFFFFF
	};

	enum PrimitiveTopology {
		PRIMITIVE_TOPOLOGY_POINT_LIST = 0,
		PRIMITIVE_TOPOLOGY_LINE_LIST = 1,
		PRIMITIVE_TOPOLOGY_LINE_STRIP = 2,
		PRIMITIVE_TOPOLOGY_TRIANGLE_LIST = 3,
		PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP = 4,
		PRIMITIVE_TOPOLOGY_TRIANGLE_FAN = 5,
		PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY = 6,
		PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY = 7,
		PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY = 8,
		PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY = 9,
		PRIMITIVE_TOPOLOGY_PATCH_LIST = 10,
		PRIMITIVE_TOPOLOGY_MAX_ENUM = 0x7FFFFFFF
	};

	class GraphicsCommand;
	class GraphicsPipeline;
	class VulkanWindow;
	class CHAOS_API VulkanPainter
	{
	public:
		friend class VulkanWindowImpl;

		virtual Ptr<VulkanWindow> CreateWindow(const std::wstring& name, uint32 width, uint32 height) = 0;
		virtual void Draw(const std::vector<Point>& pts, const std::vector<Color>& colors, const std::vector<uint16>& ind) = 0;
		//virtual void Draw(const Ptr<VulkanWindow>& window, const std::vector<Point>& pts, const std::vector<Color>& colors, const std::vector<uint16>& ind) = 0;

		virtual ~VulkanPainter();

		static Ptr<VulkanPainter> Create(const File& vert, const File& frag, uint32 device_index = GetDefaultGPUIndex());

		int front_face = FRONT_FACE_CLOCKWISE; // see enum FrontFace
		int polygon_mode = POLYGON_MODE_LINE; // see enum PolygonMode
		int topoloty = PRIMITIVE_TOPOLOGY_LINE_LIST; // see enum PrimitiveTopology

	protected:
		Ptr<VulkanWindow> window;
		//virtual void CreatePipeline(int format) = 0;

		GraphicsPipeline* pipeline;
		GraphicsCommand* command;

		std::vector<uint32> vert_spv;
		std::vector<uint32> frag_spv;

		//uint32 width; // from vulkan window
		//uint32 height; // from vulkan window

		//uint32 buffers_count;
		//void* frame_buffers;

		VulkanAllocator* allocator;
		VulkanAllocator* staging_allocator;
	};

	
	//class CHAOS_API VulkanPainter
	//{
	//public:
	//	VulkanPainter(const VulkanDevice* vkdev);
	//	virtual ~VulkanPainter();
	//	virtual void LoadModule(const File& vert, const File frag);

	//	virtual void CreatePipeline(VkFormat format, VkExtent2D extent);

	//	virtual void Draw(const std::vector<VkFramebuffer>& frame_buffers,
	//		const std::vector<Point>& pts, const std::vector<Color>& colors, const std::vector<uint16>& ind);

	//	const VulkanDevice* vkdev;
	//	std::vector<uint32> vert_data;
	//	std::vector<uint32> frag_data;

	//	GraphicsPipeline* pipeline;

	//	VkFrontFace front_face;
	//	VkPolygonMode polygon_mode;

	//private:
	//	VkFormat format;
	//	VkExtent2D extent;

	//	VulkanTensor vertex;
	//	VulkanTensor indice;
	//	std::vector<VulkanTensor> uniform;
	//	VulkanAllocator* allocator;
	//};
}