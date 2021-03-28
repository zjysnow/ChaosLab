#pragma once

#include "core/core.hpp"
#include "core/gpu.hpp"
#include "core/command.hpp"
#include "core/pipeline.hpp"

namespace chaos
{
	// create pipeline
	// setting
	// cmd.RecordPipeline
	class CHAOS_API VulkanPainter
	{
	public:
		VulkanPainter(const VulkanDevice* vkdev);
		virtual ~VulkanPainter();
		virtual void LoadModule(const File& vert, const File frag);

		virtual void CreatePipeline();

		virtual void Draw(const std::vector<Point>& pts, const std::vector<Color>& colors, const std::vector<uint16>& ind, GraphicsCommand& cmd) const;

		const VulkanDevice* vkdev;
		std::vector<uint32> vert_data;
		std::vector<uint32> frag_data;

		GraphicsPipeline* pipeline;

		VkFormat format;
		VkExtent2D extent;
		VkFrontFace front_face;
		VkPolygonMode polygon_mode;
	};
}