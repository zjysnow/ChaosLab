#include "dnn/layers/vulkan/gemm_vulkan.hpp"
#include "dnn/layers/vulkan/shaders/gemm_spv_data.hex.hpp"

namespace chaos
{
	inline namespace dnn
	{
		GEMMVulkan::GEMMVulkan() : GEMM() 
		{
			support_vulkan = true;
		}

		void GEMMVulkan::CreatePipeline(const Option& opt)
		{
			std::vector<VulkanSpecializationType> specializations;
			gemm_pipeline = new ComputePipeline(vkdev);
			gemm_pipeline->SetOptimalLocalSize(std::min(32, 1), std::min(32, 1), 1);
			gemm_pipeline->Create(gemm_spv_data, sizeof(gemm_spv_data), "gemm", specializations, 3, 10);
		}

		void GEMMVulkan::DestroyPipeline(const Option& opt)
		{
			delete gemm_pipeline;
			gemm_pipeline = nullptr;
		}

		void GEMMVulkan::Forward(const std::vector<VulkanTensor>& bottom_blobs, std::vector<VulkanTensor>& top_blobs, ComputeCommand& cmd, const Option& opt) const
		{
			CHECK_LE(2, bottom_blobs.size()) << "layer '" << type << "' expect 2 inputs at least but got " << bottom_blobs.size();
			CHECK_GE(3, bottom_blobs.size()) << "layer '" << type << "' expect 3 inputs at most but got " << bottom_blobs.size();
			const VulkanTensor& A = bottom_blobs[0];
			const VulkanTensor& B = bottom_blobs[1];

			CHECK_EQ(2, A.shape.size()) << "input A must be a matrix";
			CHECK_EQ(2, B.shape.size()) << "input B must be a matrix";

			uint32 m = A.shape[0];
			uint32 n = A.shape[1];
			uint32 p = B.shape[0];
			uint32 k = B.shape[1];

			if (transA) std::swap(m, n);
			if (transB) std::swap(p, k);

			CHECK_EQ(n, p) << "cols of A shoule be same with rows of B";

			CHECK_EQ(1, top_blobs.size()) << "layer '" << type << "' expect 1 output but got " << top_blobs.size();
			VulkanTensor& C = top_blobs[0];
			if (C.empty()) C.Create(Shape(m, k), Steps(k, 1), Depth::D4, Packing::CHW, opt.blob_vkallocator);
			CHECK_EQ(Shape(m, k), C.shape);

			uint32 astep = A.steps[0];
			uint32 bstep = B.steps[0];
			uint32 cstep = C.steps[0];

			if (bottom_blobs.size() == 3 && beta != 0)
			{
				CHECK(not bottom_blobs[2].empty()) << "really?";
				CHECK_EQ(2, bottom_blobs[2].shape.size()) << "input C must be a matrix";
				CHECK_EQ(bottom_blobs[2].shape, C.shape) << "shape of input C shoule be " << m << "x" << k;
				cmd.RecordClone(bottom_blobs[2], C, opt);
			}

			std::vector<VulkanTensor> bindings(3);
			bindings[0] = A;
			bindings[1] = B;
			bindings[2] = C;

			std::vector<VulkanConstantType> constants(10);
			constants[0].i = m;
			constants[1].i = n;
			constants[2].i = k;
			constants[3].i = astep;
			constants[4].i = bstep;
			constants[5].i = cstep;
			constants[6].i = transA ? 1 : 0;
			constants[7].i = transB ? 1 : 0;
			constants[8].f = alpha;
			constants[9].f = bottom_blobs.size() == 2 ? 0 : beta;

			Shape dispatcher(1, m, k);

			cmd.RecordPipeline(gemm_pipeline, bindings, constants, dispatcher);
		}
	}
}