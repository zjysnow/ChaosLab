#include "core/gpu.hpp"
#include "core/buffer.hpp"
#include "core/tensor.hpp"
#include "core/command.hpp"
#include "core/pipeline.hpp"
#include "core/allocator.hpp"

#include "highgui/painter.hpp"

#include <fstream>

namespace chaos
{
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
		VulkanPainterImpl(const File& vert, const File& frag, const VulkanDevice* vkdev)
		{
			auto Read = [](const File& file) {

				std::wfstream fs(file, std::ios::in | std::ios::binary | std::ios::ate);
				CHECK(fs.is_open()) << "can't not open file " << file.name();
				size_t file_size = fs.tellg();
				AutoBuffer<uint16> buffer(file_size);
				fs.seekg(0);
				fs.read((wchar_t*)buffer.data(), file_size);
				fs.close();

				std::vector<uint32> spv(file_size / sizeof(uint32));
				// convert int16 to int8
				uint8* data = (uint8*)spv.data();
				for (size_t i = 0; i < file_size; i++)
				{
					data[i] = (uint8)buffer[i];
				}
				return spv;
			};

			vert_spv = Read(vert);
			frag_spv = Read(frag);

			command = new GraphicsCommand(vkdev);
			pipeline = new GraphicsPipeline(vkdev);

			allocator = new VulkanAllocator(vkdev);
			staging_allocator = new VulkanStagingAllocator(vkdev);
		}

		~VulkanPainterImpl() 
		{
			
		}

		void CreatePipeline(int format) final
		{
			pipeline->Create(vert_spv.data(), vert_spv.size() * sizeof(uint32), frag_spv.data(), frag_spv.size() * sizeof(uint32), 
				(VkFormat)format, width, height, (VkPolygonMode)polygon_mode, (VkFrontFace)front_face, (VkPrimitiveTopology)topoloty);
		}

		void Draw(const std::vector<Point>& pts, const std::vector<Color>& colors, const std::vector<uint16>& ind) final
		{
			command->Init(buffers_count);

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

			uniform.resize(buffers_count);
			for (uint32 i = 0; i < buffers_count; i++)
			{
				uniform[i] = VulkanTensor(Shape(4u * 4u * 3), DataType::D4, Packing::CHW, staging_allocator);
			}

			VulkanTensor vertex_staging, indices_staging;
			command->RecordUpload(vertex_data, vertex_staging, staging_allocator);
			command->RecordUpload(indices_data, indices_staging, staging_allocator);

			command->RecordClone(vertex_staging, vertex, allocator);
			command->RecordClone(indices_staging, indices, allocator);

			command->RecordPipeline(pipeline, buffers_count, (VkFramebuffer*)frame_buffers, width, height, vertex, indices, uniform);
		}

		VulkanTensor vertex;
		VulkanTensor indices;
		std::vector<VulkanTensor> uniform;
	};

	Ptr<VulkanPainter> VulkanPainter::Create(const File& vert, const File& frag, uint32 device_index)
	{
		return Ptr<VulkanPainter>(new VulkanPainterImpl(vert, frag, GetGPUDevice(device_index)));
	}

	//VulkanPainter::VulkanPainter(const VulkanDevice* vkdev) : vkdev(vkdev) 
	//{
	//	allocator = new VulkanAllocator(vkdev);
	//	pipeline = new GraphicsPipeline(vkdev);
	//	//cmd = new GraphicsCommand(vkdev);
	//}
	//VulkanPainter::~VulkanPainter() { delete pipeline; }

	//void VulkanPainter::LoadModule(const File& vert, const File frag)
	//{
	//	auto Read = [](const File& file) {
	//		std::fstream fs((std::string)file, std::ios::in | std::ios::binary);

	//		CHECK(fs.is_open()) << "can't not open file " << file.name();

	//		size_t file_size = fs.tellg();
	//		std::vector<uint32> buffer(file_size / sizeof(uint32));

	//		fs.seekg(0);
	//		fs.read((char*)buffer.data(), file_size);

	//		fs.close();

	//		return buffer;
	//	};

	//	vert_data = Read(vert);
	//	frag_data = Read(frag);
	//}

	//void VulkanPainter::CreatePipeline(VkFormat format, VkExtent2D extent)
	//{
	//	pipeline->Create(vert_data.data(), vert_data.size() * sizeof(uint32), frag_data.data(), frag_data.size() * sizeof(uint32),
	//		format, extent, polygon_mode);
	//}

	//void VulkanPainter::Draw(const std::vector<VkFramebuffer>& frame_buffers, 
	//	const std::vector<Point>& pts, const std::vector<Color>& colors, const std::vector<uint16>& ind)
	//{
	//	size_t n = pts.size();
	//	// re-alignment
	//	AutoBuffer<float> buffer(n * 5);
	//	for (int i = 0; i < n; i++)
	//	{
	//		float* data = buffer.data() + i * 5LL;
	//		data[0] = pts[i].x;
	//		data[1] = pts[i].y;
	//		data[2] = colors[i].r;
	//		data[3] = colors[i].g;
	//		data[4] = colors[i].b;
	//	}

	//	Tensor vertex_data = Tensor(Shape((uint32)buffer.size()), DataType::D4, Packing::CHW, buffer.data());
	//	Tensor indice_data = Tensor(Shape((uint32)ind.size()), DataType::D2, Packing::CHW, const_cast<uint16*>(ind.data()));

	//	VulkanAllocator* staging_allocator = new VulkanStagingAllocator(vkdev);
	//	{
	//		VulkanTensor vertex_staging;
	//		VulkanTensor indice_staging;

	//		//cmd->RecordUpload(vertex_data, vertex_staging, staging_allocator);
	//		//cmd->RecordUpload(indice_data, indice_staging, staging_allocator);

	//		//cmd->RecordClone(vertex_staging, vertex, allocator);
	//		//cmd->RecordClone(indice_staging, indice, allocator);
	//	}

	//	//cmd->RecordPipeline(pipeline, frame_buffers, extent, vertex, indice, uniform);
	//}
}