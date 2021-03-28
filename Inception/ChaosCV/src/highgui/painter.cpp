#include "core/tensor.hpp"
#include "core/gpu.hpp"
#include "core/pipeline.hpp"
#include "highgui/painter.hpp"


#include <fstream>

namespace chaos
{
	VulkanPainter::VulkanPainter(const VulkanDevice* vkdev) : vkdev(vkdev) {}
	VulkanPainter::~VulkanPainter() { delete pipeline; }

	void VulkanPainter::LoadModule(const File& vert, const File frag)
	{
		auto Read = [](const File& file) {
			std::fstream fs((std::string)file, std::ios::in | std::ios::binary);

			CHECK(fs.is_open()) << "can't not open file " << file.name();

			size_t file_size = fs.tellg();
			std::vector<uint32> buffer(file_size / sizeof(uint32));

			fs.seekg(0);
			fs.read((char*)buffer.data(), file_size);

			fs.close();

			return buffer;
		};

		vert_data = Read(vert);
		frag_data = Read(frag);
	}

	void VulkanPainter::CreatePipeline()
	{
		pipeline = new GraphicsPipeline(vkdev);

		pipeline->Create(vert_data.data(), vert_data.size() * sizeof(uint32), frag_data.data(), frag_data.size() * sizeof(uint32),
			format, extent, polygon_mode);
	}

	void VulkanPainter::Draw(const std::vector<Point>& pts, const std::vector<Color>& colors, const std::vector<uint16>& ind, GraphicsCommand& cmd) const
	{
		size_t n = pts.size();
		// re-alignment
		AutoBuffer<float> buffer(n * 5);
		for (int i = 0; i < n; i++)
		{
			float* data = buffer.data() + i * 5LL;
			data[0] = pts[i].x;
			data[1] = pts[i].y;
			data[2] = colors[i].r;
			data[3] = colors[i].g;
			data[4] = colors[i].b;
		}

		Tensor vertex_data = Tensor(Shape((uint32)buffer.size()), DataType::D4, Packing::CHW, buffer.data());
		Tensor indices_data = Tensor(Shape((uint32)ind.size()), DataType::D2, Packing::CHW, const_cast<uint16*>(ind.data()));

		VulkanAllocator* staging_allocator = new VulkanAllocator(vkdev);
		{
			VulkanTensor vertex_staging;
			VulkanTensor indice_staging;
		}
		delete staging_allocator;


		//cmd.RecordPipeline(pipeline, frame_buffers);
	}
}