#include "dnn/layers/vulkan/reduce_vulkan.hpp"
#include "dnn/layers/vulkan/shaders/reduce_spv_data.hex.hpp"

namespace chaos
{
	inline namespace dnn
	{
		ReduceVulkan::ReduceVulkan() : Reduce()
		{
			support_vulkan = true;
		}

		void ReduceVulkan::CreatePipeline(const Option& opt)
		{
			std::vector<VulkanSpecializationType> specializations;
			reduce_pipeline = new ComputePipeline(vkdev);
			reduce_pipeline->SetOptimalLocalSize(std::min(64, 1), 1, 1);
			reduce_pipeline->Create(reduce_spv_data, sizeof(reduce_spv_data), "reduce", specializations, 7, 5);
		}
		void ReduceVulkan::DestroyPipeline(const Option& opt)
		{
			delete reduce_pipeline;
			reduce_pipeline = nullptr;
		}

		void ReduceVulkan::Forward(const std::vector<VulkanTensor>& bottom_blobs, std::vector<VulkanTensor>& top_blobs, ComputeCommand& cmd, const Option& opt) const
		{
			CHECK_EQ(1, bottom_blobs.size()) << "layer Sum expect 1 input but got " << bottom_blobs.size();
			CHECK_EQ(1, top_blobs.size()) << "layer Sum expect 1 output but got " << top_blobs.size();

			const VulkanTensor& A = bottom_blobs[0];
			int dims = (int)A.shape.size();
			CHECK_LE(vecdim.size(), dims) << "dims of A should greater-equal than size of vecdims";

			Shape shape = A.shape;
			Shape inverse = Array<uint32>(1, dims);
			float N = 1.f;
			if (all)
			{
				for (int i = 0; i < dims; i++)
				{
					inverse[i] = shape[i];
					N *= shape[i];
					shape[i] = 1;
				}
			}
			else
			{
				for (const auto& i : vecdim)
				{
					CHECK_LE(i, dims) << "out of range";
					inverse[i] = shape[i];
					N *= shape[i];
					shape[i] = 1;
				}
			}
			Steps steps = shape.steps();

			VulkanTensor& S = top_blobs[0];
			if (S.empty()) S.Create(shape, steps, Depth::D4, Packing::CHW, opt.blob_vkallocator);
			CHECK_EQ(shape, S.shape);

			std::vector<VulkanTensor> bindings(7);
			bindings[0] = A;
			bindings[1] = A.shape.Upload(opt);
			bindings[2] = A.steps.Upload(opt);

			bindings[3] = S;
			bindings[4] = S.shape.Upload(opt);
			bindings[5] = S.steps.Upload(opt);

			bindings[6] = inverse.Upload(opt);

			int total = shape.total();
			std::vector<VulkanConstantType> constants(5);
			constants[0].i = total;
			constants[1].i = dims;
			constants[2].i = inverse.total();
			constants[3].i = (int)op_type;
			constants[4].f = op_type == AVG ? alpha / N : alpha;

			Shape dispatcher(1, 1, total);

			cmd.RecordPipeline(reduce_pipeline, bindings, constants, dispatcher);
		}
	}
}