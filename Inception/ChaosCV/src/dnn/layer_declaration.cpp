#include "dnn/layer.hpp"
#include "dnn/option.hpp"
#include "dnn/layer_factory.hpp"

#include "dnn/layers/abs.hpp"
#include "dnn/layers/vulkan/abs_vulkan.hpp"
namespace chaos::inline dnn
{
	class AbsFinal : virtual public AbsVulkan
	{
	public:
		void CreatePipeline(const Option& opt) final
		{
			Abs::CreatePipeline(opt);
			if (opt.use_vulkan_compute) AbsVulkan::CreatePipeline(opt);
		}
		void DestroyPipeline(const Option& opt) final
		{
			if (opt.use_vulkan_compute) AbsVulkan::DestroyPipeline(opt);
			Abs::DestroyPipeline(opt);
		}
	};
	REGISTER_LAYER("Abs", AbsFinal);
}

#include "dnn/layers/backsubst.hpp"
namespace chaos::inline dnn
{
	REGISTER_LAYER("Backsubst", Backsubst);
}

#include "dnn/layers/binary_op.hpp"
#include "dnn/layers/vulkan/binary_op_vulkan.hpp"
namespace chaos::inline dnn
{
	class BinaryOpFinal : virtual public BinaryOpVulkan
	{
	public:
		void CreatePipeline(const Option& opt) final
		{
			BinaryOp::CreatePipeline(opt);
			if (opt.use_vulkan_compute) BinaryOpVulkan::CreatePipeline(opt);
		}
		void DestroyPipeline(const Option& opt) final
		{
			if (opt.use_vulkan_compute) BinaryOpVulkan::DestroyPipeline(opt);
			BinaryOp::DestroyPipeline(opt);
		}
	};
	REGISTER_LAYER("BinaryOp", BinaryOpFinal);
}

#include "dnn/layers/diag.hpp"
namespace chaos::inline dnn
{
	REGISTER_LAYER("Diag", Diag);
}

#include "dnn/layers/gemm.hpp"
#include "dnn/layers/vulkan/gemm_vulkan.hpp"
namespace chaos::inline dnn
{
	class GEMMFinal : public GEMMVulkan
	{
	public:
		void CreatePipeline(const Option& opt) final
		{
			GEMM::CreatePipeline(opt);
			if (opt.use_vulkan_compute) GEMMVulkan::CreatePipeline(opt);
		}
		void DestroyPipeline(const Option& opt) final
		{
			if (opt.use_vulkan_compute) GEMMVulkan::DestroyPipeline(opt);
			GEMM::DestroyPipeline(opt);
		}
	};
	REGISTER_LAYER("GEMM", GEMMFinal);
}

#include "dnn/layers/invert.hpp"
namespace chaos::inline dnn
{
	REGISTER_LAYER("Invert", Invert);
}

#include "dnn/layers/permute.hpp"
#include "dnn/layers/vulkan/permute_vulkan.hpp"
namespace chaos::inline dnn
{
	class PermuteFinal : public PermuteVulkan
	{
	public:
		void CreatePipeline(const Option& opt) final
		{
			Permute::CreatePipeline(opt);
			if (opt.use_vulkan_compute) PermuteVulkan::CreatePipeline(opt);
		}
		void DestroyPipeline(const Option& opt) final
		{
			if (opt.use_vulkan_compute) PermuteVulkan::DestroyPipeline(opt);
			Permute::DestroyPipeline(opt);
		}
	};
	REGISTER_LAYER("Permute", PermuteFinal);
}

#include "dnn/layers/reduce.hpp"
#include "dnn/layers/vulkan/reduce_vulkan.hpp"
namespace chaos::inline dnn
{
	class ReduceFinal : public ReduceVulkan
	{
	public:
		void CreatePipeline(const Option& opt) final
		{
			Reduce::CreatePipeline(opt);
			if (opt.use_vulkan_compute) ReduceVulkan::CreatePipeline(opt);
		}
		void DestroyPipeline(const Option& opt) final
		{
			if (opt.use_vulkan_compute) ReduceVulkan::DestroyPipeline(opt);
			Reduce::DestroyPipeline(opt);
		}
	};
	REGISTER_LAYER("Reduce", ReduceFinal);
}

#include "dnn/layers/svd.hpp"
namespace chaos::inline dnn
{
	REGISTER_LAYER("SVD", SVD);
}

#include "dnn/layers/var.hpp"
namespace chaos::inline dnn
{
	REGISTER_LAYER("Var", Var);
}