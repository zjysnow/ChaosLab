#pragma once

#include "core/core.hpp"

namespace chaos
{
	// create pipeline
	// setting
	// cmd.RecordPipeline
	enum FrontFace
	{

	};

	enum PolygonMode
	{
		POLYGON_MODE_FILL = 0,
		POLYGON_MODE_LINE = 1,
		POLYGON_MODE_POINT = 2,
		POLYGON_MODE_FILL_RECTANGLE_NV = 1000153000,
		POLYGON_MODE_MAX_ENUM = 0x7FFFFFFF
	};

	class GraphicsCommand;
	class GraphicsPipeline;
	class CHAOS_API VulkanPainter
	{
	public:
		friend class VulkanWindowImpl;

		virtual void Draw(const std::vector<Point>& pts, const std::vector<Color>& colors, const std::vector<uint16>& ind) = 0;

		virtual ~VulkanPainter();

		static Ptr<VulkanPainter> Create(const File& vert, const File& frag, uint32 device_index = GetDefaultGPUIndex());

		int front_face;
		int polygon_mode;

	protected:
		virtual void CreatePipeline(int format) = 0;

		GraphicsPipeline* pipeline;
		GraphicsCommand* command;

		std::vector<uint32> vert_spv;
		std::vector<uint32> frag_spv;

		uint32 width; // from vulkan window
		uint32 height; // from vulkan window

		uint32 buffers_count;
		void* frame_buffers;

		

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