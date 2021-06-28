#include "dnn/layers/vulkan/permute_vulkan.hpp"
#include "dnn/layers/vulkan/shaders/permute_spv_data.hex.hpp"
#include "dnn/layers/vulkan/shaders/transpose_spv_data.hex.hpp"

namespace chaos
{
	inline namespace dnn
	{
		PermuteVulkan::PermuteVulkan() : Permute() 
		{
			support_vulkan = true;
		}

		void PermuteVulkan::CreatePipeline(const Option& opt)
		{
			permute_pipeline = new ComputePipeline(vkdev);
			{
				std::vector<VulkanSpecializationType> specializations;
				permute_pipeline->SetOptimalLocalSize(std::min(64, 1), 1, 1);
				permute_pipeline->Create(permute_spv_data, sizeof(permute_spv_data), "permute", specializations, 7, 2);
			}

			transpose_pipeline = new ComputePipeline(vkdev);
			{
				std::vector<VulkanSpecializationType> specializations;
				transpose_pipeline->SetOptimalLocalSize(std::min(32, 1), std::min(32, 1), 1);
				transpose_pipeline->Create(transpose_spv_data, sizeof(transpose_spv_data), "transpose", specializations, 2, 4);
			}
		}
		void PermuteVulkan::DestroyPipeline(const Option& opt)
		{
			delete permute_pipeline;
			permute_pipeline = nullptr;

			delete transpose_pipeline;
			transpose_pipeline = nullptr;
		}

		void PermuteVulkan::Forward(const std::vector<VulkanTensor>& bottom_blobs, std::vector<VulkanTensor>& top_blobs, ComputeCommand& cmd, const Option& opt) const
		{
			CHECK_EQ(1, bottom_blobs.size()) << "layer 'Permute' expect 1 input but got " << bottom_blobs.size();
			const VulkanTensor& A = bottom_blobs[0];
			int dims = (int)A.shape.size();
			CHECK_EQ(orders.size(), dims) << "num axes expect " << orders.size() << " but got " << dims;

			bool need_permute = false;
			for (int i = 0; i < dims; i++)
			{
				if (i != orders[i])
				{
					need_permute = true;
					break;
				}
			}

			CHECK_EQ(1, top_blobs.size()) << "layer 'Permute' expect 1 output but got " << top_blobs.size();
			VulkanTensor& P = top_blobs[0];
			Shape shape = A.shape;
			if (not need_permute)
			{
				if (P.empty()) P.Create(shape, shape.steps(), Depth::D4, Packing::CHW, opt.blob_vkallocator);
				CHECK_EQ(shape, P.shape) << "expect " << shape << " but got " << P.shape;
				cmd.RecordClone(A, P, opt);
			}
			else
			{
				for (int i = 0; i < dims; i++) shape[i] = A.shape[orders[i]];
				if (P.empty()) P.Create(shape, shape.steps(), Depth::D4, Packing::CHW, opt.blob_vkallocator);
				CHECK_EQ(shape, P.shape) << "expect " << shape << " but got " << P.shape;

				int dims = (int)shape.size();
				if (dims == 2)
				{
					std::vector<VulkanTensor> bindings(2);
					bindings[0] = A;
					bindings[1] = P;

					std::vector<VulkanConstantType> constants(4);
					constants[0].i = shape[1]; // w
					constants[1].i = shape[0]; // h
					constants[2].i = A.steps[0]; // astep
					constants[3].i = P.steps[0]; // bstep

					Shape dispatcher(1, shape[0], shape[1]);

					cmd.RecordPipeline(transpose_pipeline, bindings, constants, dispatcher);
				}
				else
				{
					std::vector<VulkanTensor> bindings(7);
					bindings[0] = A;
					bindings[1] = A.shape.Upload(opt);
					bindings[2] = A.steps.Upload(opt);

					bindings[3] = P;
					bindings[4] = P.shape.Upload(opt);
					bindings[5] = P.steps.Upload(opt);

					bindings[6] = orders.Upload(opt);

					int total = shape.total();
					int dims = (int)shape.size();
					std::vector<VulkanConstantType> constants(2);
					constants[0].i = total;
					constants[1].i = dims;

					Shape dispatcher(1, 1, total);

					cmd.RecordPipeline(permute_pipeline, bindings, constants, dispatcher);
				}
			}
		}
	}
}