#include "dnn/layers/vulkan/binary_op_vulkan.hpp"
#include "dnn/layers/vulkan/shaders/binary_op_spv_data.hex.hpp"

namespace chaos
{
	inline namespace dnn
	{
		BinaryOpVulkan::BinaryOpVulkan() : BinaryOp()
		{
			support_vulkan = true;
		}

		void BinaryOpVulkan::CreatePipeline(const Option& opt)
		{
			std::vector<VulkanSpecializationType> specializations;
			binary_op_pipeline = new ComputePipeline(vkdev);
			binary_op_pipeline->SetOptimalLocalSize(std::min(64, 1), 1, 1);
			binary_op_pipeline->Create(binary_op_spv_data, sizeof(binary_op_spv_data), "binary_op", specializations, 9, 3);
		}
		void BinaryOpVulkan::DestroyPipeline(const Option& opt)
		{
			delete binary_op_pipeline;
			binary_op_pipeline = nullptr;
		}

		void BinaryOpVulkan::Forward(const std::vector<VulkanTensor>& bottom_blobs, std::vector<VulkanTensor>& top_blobs, ComputeCommand& cmd, const Option& opt) const
		{
			VulkanTensor a = bottom_blobs[0];
			VulkanTensor b = bottom_blobs[1];

			int a_dims = (int)a.shape.size();
			int b_dims = (int)b.shape.size();
			// expand the shape to the same dims
			int a_cnt = std::max(0, b_dims - a_dims);
			int b_cnt = std::max(0, a_dims - b_dims);
			a.steps.Expand(0, a_cnt, (uint32)a.total());
			a.shape.Expand(0, a_cnt);
			b.steps.Expand(0, b_cnt, (uint32)b.total());
			b.shape.Expand(0, b_cnt);

			CHECK_EQ(1, top_blobs.size()) << "layer 'BinaryOp' expcet 1 output but got " << top_blobs.size();
			VulkanTensor& c = top_blobs[0];
			Shape c_shape = a.shape & b.shape; //Broadcast(a.shape, b.shape);
			if (c.empty()) c.Create(c_shape, c_shape.steps(), Depth::D4, Packing::CHW, opt.blob_vkallocator);
			CHECK_EQ(c_shape, c.shape) << "expect " << c_shape << " but got " << c.shape;

			std::vector<VulkanTensor> bindings(9);
			bindings[0] = a;
			bindings[1] = a.shape.Upload(opt);
			bindings[2] = a.steps.Upload(opt);

			bindings[3] = b;
			bindings[4] = b.shape.Upload(opt);
			bindings[5] = b.steps.Upload(opt);

			bindings[6] = c;
			bindings[7] = c.shape.Upload(opt);
			bindings[8] = c.steps.Upload(opt);

			int total = c_shape.total();
			int dims = c_shape.size();
			std::vector<VulkanConstantType> constants(3);
			constants[0].i = total;
			constants[1].i = dims;
			constants[2].i = op_type;

			Shape dispatcher(1, 1, total);

			cmd.RecordPipeline(binary_op_pipeline, bindings, constants, dispatcher);
		}
	}
}