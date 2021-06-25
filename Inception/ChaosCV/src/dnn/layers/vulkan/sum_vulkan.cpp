#include "dnn/layers/vulkan/sum_vulkan.hpp"
#include "dnn/layers/vulkan/shaders/sum_spv_data.hex.hpp"

namespace chaos
{
	inline namespace dnn
	{
		SumVulkan::SumVulkan() : Sum() 
		{
			support_vulkan = true;
		}

		void SumVulkan::CreatePipeline(const Option& opt)
		{
			std::vector<VulkanSpecializationType> specializations;
			sum_pipeline = new ComputePipeline(vkdev);
			sum_pipeline->SetOptimalLocalSize(std::min(64, 1), 1, 1);
			sum_pipeline->Create(sum_spv_data, sizeof(sum_spv_data), "sum", specializations, 7, 3);
		}
		void SumVulkan::DestroyPipeline(const Option& opt)
		{
			delete sum_pipeline;
			sum_pipeline = nullptr;
		}

		void SumVulkan::Forward(const std::vector<VulkanTensor>& bottom_blobs, std::vector<VulkanTensor>& top_blobs, ComputeCommand& cmd, const Option& opt) const
		{
			CHECK_EQ(1, bottom_blobs.size()) << "layer Sum expect 1 input but got " << bottom_blobs.size();
			CHECK_EQ(1, top_blobs.size()) << "layer Sum expect 1 output but got " << top_blobs.size();

			const VulkanTensor& A = bottom_blobs[0];
			int dims = (int)A.shape.size();
			CHECK_LE(vecdim.size(), dims) << "dims of A should greater-equal than size of vecdims";

			Shape shape = A.shape;
			Shape inverse = Array<uint32>(1, dims);
			if (all)
			{
				for (int i = 0; i < dims; i++)
				{
					inverse[i] = shape[i];
					shape[i] = 1;
				}
			}
			else
			{
				for (const auto& i : vecdim)
				{
					CHECK_LE(i, dims) << "out of range";
					inverse[i] = shape[i];
					shape[i] = 1;
				}
			}
			Steps steps = shape.steps();

			VulkanTensor& S = top_blobs[0];
			if (S.empty()) S.Create(shape, steps, Depth::D4, Packing::CHW, opt.blob_vkallocator);
			CHECK_EQ(shape, S.shape);
			//memset(S.data, 0, shape.total() * sizeof(float));

			std::vector<VulkanTensor> bindings(7);
			bindings[0] = A;
			bindings[1] = A.shape.Upload(opt);
			bindings[2] = A.steps.Upload(opt);

			bindings[3] = S;
			bindings[4] = S.shape.Upload(opt);
			bindings[5] = S.steps.Upload(opt);

			bindings[6] = inverse.Upload(opt);

			int total = shape.total();
			std::vector<VulkanConstantType> constants(3);
			constants[0].i = total;
			constants[1].i = dims;
			constants[2].i = inverse.total();

			Shape dispatcher(1, 1, total);

			cmd.RecordPipeline(sum_pipeline, bindings, constants, dispatcher);
		}
	}
}