#include "dnn/layers/vulkan/permute_vulkan.hpp"
#include "dnn/layers/vulkan/shaders/permute_spv_data.hex.hpp"

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
			std::vector<VulkanSpecializationType> specializations;
			permute_pipeline = new ComputePipeline(vkdev);
			permute_pipeline->SetOptimalLocalSize(std::min(64, 1), 1, 1);
			permute_pipeline->Create(permute_spv_data, sizeof(permute_spv_data), "permute", specializations, 5, 2);
		}
		void PermuteVulkan::DestroyPipeline(const Option& opt)
		{
			delete permute_pipeline;
			permute_pipeline = nullptr;
		}

		void PermuteVulkan::Forward(const std::vector<VulkanTensor>& bottom_blobs, std::vector<VulkanTensor>& top_blobs, ComputeCommand& cmd, const Option& opt) const
		{
			CHECK_EQ(1, bottom_blobs.size()) << "layer 'Permute' expect 1 input but got " << bottom_blobs.size();
			const VulkanTensor& A = bottom_blobs[0];
			size_t dims = A.shape.size();
			CHECK_EQ(orders.size(), dims) << "num axes expect " << orders.size() << " but got " << dims; // Format("num axes expect %d, but got %d", orders.size(), dims);

			bool need_permute = false;
			for (size_t i = 0; i < dims; i++)
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
				cmd.RecordClone(A, P, opt);
			}
			else
			{
				for (size_t i = 0; i < dims; i++) shape[i] = A.shape[orders[i]];
				if (P.empty()) P.Create(shape, shape.steps(), Depth::D4, Packing::CHW, opt.blob_vkallocator);
				CHECK_EQ(shape, P.shape) << "expect " << shape << " but got " << P.shape;

				std::vector<VulkanConstantType> constants(2);
				constants[0].i = dims; // bottom_top_blobs[0].shape[0];
				constants[1].i = (int)shape.total(); // bottom_top_blobs[0].shape[1];

				Shape dispatcher(1, 1, shape.total());

				Option staging_opt;
				staging_opt.blob_vkallocator = opt.staging_vkallocator;
				staging_opt.staging_vkallocator = opt.staging_vkallocator;

				VulkanTensor a_steps;
				VulkanTensor p_steps;
				VulkanTensor vk_orders;

				cmd.RecordUpload(A.steps, a_steps, staging_opt);
				cmd.RecordUpload(P.steps, p_steps, staging_opt);
				cmd.RecordUpload(orders, vk_orders, staging_opt);

				std::vector<VulkanTensor> bindings(5);
				bindings[0] = A;
				bindings[1] = a_steps;
				bindings[2] = P;
				bindings[3] = p_steps;
				bindings[4] = vk_orders;

				cmd.RecordPipeline(permute_pipeline, bindings, constants, dispatcher);
			}
		}
	}
}