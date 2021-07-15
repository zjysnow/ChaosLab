#include "highgui/painter.hpp"
#include "highgui/window.hpp"

#include "core/tensor.hpp"
#include "core/file.hpp"
#include "core/gpu.hpp"
#include "core/command.hpp"
#include "core/pipeline.hpp"

#include "dnn/option.hpp"

#include <vulkan/vulkan.hpp>

#include <fstream>

namespace chaos
{
	Painter::Painter(const VulkanDevice* vkdev) : vkdev(vkdev)
	{
		pipeline = new GraphicsPipeline(vkdev);
		command = new GraphicsCommand(vkdev);

		allocator = new VulkanLocalAllocator(vkdev);
		staging_allocator = new VulkanStagingAllocator(vkdev);
	}

	Painter::~Painter() 
	{
		delete allocator;
		delete staging_allocator;

		delete command;
		delete pipeline;
	}

	class VulkanPainter : public Painter
	{
	public:
		VulkanPainter(const File& vert, const File& frag,const VulkanDevice* vkdev) : Painter(vkdev)
		{
			auto Read = [](const File& file) {
				std::fstream fs(file.data(), std::ios::in | std::ios::binary | std::ios::ate);
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

		Ptr<Window> CreateWindow(const std::wstring& name, uint32 width, uint32 height)
		{
			window = Window::Create(name, width, height, vkdev);

			extent.height = window->height();
			extent.width = window->width();
			buffers_count = window->image_count;

			pipeline->Create(vert_spv.data(), vert_spv.size() * sizeof(uint32), frag_spv.data(), frag_spv.size() * sizeof(uint32),
				extent, (VkFormat)window->format(),
				(VkPolygonMode)polygon_mode, (VkFrontFace)front_face,
				(VkPrimitiveTopology)topoloty, (VkCullModeFlagBits)cull_mode);

			command->Create(buffers_count);

			frame_buffers = window->CreateFrameBuffer(pipeline->render_pass);

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

			Tensor vertex_data = Tensor(Shape((int)buffer.size()), Depth::D4, Packing::CHW, buffer.data());
			Tensor indices_data = Tensor(Shape((int)ind.size()), Depth::D2, Packing::CHW, (void*)ind.data());

			// default ubo
			Tensor ubo = CreateUniformObject();
			uniform.resize(buffers_count);
			for (uint32 i = 0; i < buffers_count; i++)
			{
				command->RecordUpload(ubo, uniform[i], staging_allocator, staging_allocator);
			}

			command->RecordUpload(vertex_data, vertex, allocator, staging_allocator);
			command->RecordUpload(indices_data, indices, allocator, staging_allocator);

			command->RecordPipeline(pipeline, buffers_count, frame_buffers, extent, vertex, indices, uniform);
		}

		void UpdateUniformBuffer(uint32 image_index) final
		{
			Tensor ubo = CreateUniformObject();
			command->RecordUpload(ubo, uniform[image_index], staging_allocator, staging_allocator);
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

	Ptr<Painter> Painter::Create(const File& vert, const File& frag, uint32 device_index)
	{
		return Ptr<Painter>(new VulkanPainter(vert, frag, GetGPUDevice(device_index)));
	}
}