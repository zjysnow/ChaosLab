#include "core/gpu.hpp"
#include "core/buffer.hpp"
#include "core/tensor.hpp"
#include "core/command.hpp"
#include "core/pipeline.hpp"
#include "core/allocator.hpp"

#include "highgui/window.hpp"
#include "highgui/painter.hpp"

#include <fstream>

namespace chaos
{
	VulkanPainter::VulkanPainter(const VulkanDevice* vkdev) : vkdev(vkdev)
	{
		command = new GraphicsCommand(vkdev);
		pipeline = new GraphicsPipeline(vkdev);

		allocator = new VulkanAllocator(vkdev);
		staging_allocator = new VulkanStagingAllocator(vkdev);
	}

	VulkanPainter::~VulkanPainter() 
	{ 
		delete staging_allocator;
		delete allocator;

		delete pipeline; 
		delete command;
	}

	class VulkanPainterImpl : public VulkanPainter
	{
	public:
		VulkanPainterImpl(const File& vert, const File& frag, const VulkanDevice* vkdev) : VulkanPainter(vkdev)
		{
			auto Read = [](const File& file) {
				std::fstream fs(file, std::ios::in | std::ios::binary | std::ios::ate);
				CHECK(fs.is_open()) << "can't not open file " << file.name();
				size_t file_size = fs.tellg();

				std::vector<uint32> spv(file_size / sizeof(uint32));

				fs.seekg(0);
				fs.read((char*)spv.data(), file_size);
				fs.close();

				return spv;
			};

			vert_spv = Read(vert);
			frag_spv = Read(frag);
		}

		Ptr<VulkanWindow> CreateWindow(const std::wstring& name, uint32 width, uint32 height)
		{
			window = VulkanWindow::Create(name, width, height, vkdev);

			extent.height = window->height();
			extent.width = window->width();
			buffers_count = window->image_count;

			pipeline->Create(vert_spv.data(), vert_spv.size() * sizeof(uint32), frag_spv.data(), frag_spv.size() * sizeof(uint32),
				extent, (VkFormat)window->image_format(),
				(VkPolygonMode)polygon_mode, (VkFrontFace)front_face, 
				(VkPrimitiveTopology)topoloty, (VkCullModeFlagBits)cull_mode);

			command->Create(buffers_count);

			frame_buffers = (VkFramebuffer*)window->CreateFrameBuffer(pipeline->render_pass);

			window->painter = this;
			return window;
		}

		void Draw(const std::vector<Point>& pts, const std::vector<Color>& colors, const std::vector<uint16>& ind)
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
			Tensor indices_data = Tensor(Shape((uint32)ind.size()), DataType::D2, Packing::CHW, (void*)ind.data());

			// default ubo
			Tensor ubo = CreateUniformObject();
			uniform.resize(buffers_count);
			for (uint32 i = 0; i < buffers_count; i++)
			{
				command->RecordUpload(ubo, uniform[i], staging_allocator);
				//uniform[i] = VulkanTensor(Shape(3,4,4), DataType::D4, Packing::CHW, staging_allocator);
			}

			VulkanTensor vertex_staging, indices_staging;
			command->RecordUpload(vertex_data, vertex_staging, staging_allocator);
			command->RecordUpload(indices_data, indices_staging, staging_allocator);

			command->RecordClone(vertex_staging, vertex, allocator);
			command->RecordClone(indices_staging, indices, allocator);

			command->RecordPipeline(pipeline, buffers_count, frame_buffers, extent, vertex, indices, uniform);
		}

		void UpdateUniformBuffer(uint32 image_index) override
		{
			Tensor ubo = CreateUniformObject();
			command->RecordUpload(ubo, uniform[image_index], staging_allocator);
		}

		VkExtent2D extent;
		uint32 buffers_count;

		VulkanTensor vertex;
		VulkanTensor indices;
		std::vector<VulkanTensor> uniform;
		
		VkFramebuffer* frame_buffers;

		std::vector<uint32> vert_spv;
		std::vector<uint32> frag_spv;
	};

	Ptr<VulkanPainter> VulkanPainter::Create(const File& vert, const File& frag, uint32 device_index)
	{
		return Ptr<VulkanPainter>(new VulkanPainterImpl(vert, frag, GetGPUDevice(device_index)));
	}
}