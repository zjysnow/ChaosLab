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

#include "dnn/layers/back_subst.hpp"
namespace chaos::inline dnn
{
	REGISTER_LAYER("BackSubst", BackSubst);
}

#include "dnn/layers/binary_op.hpp"
namespace chaos::inline dnn
{
	REGISTER_LAYER("BinaryOp", BinaryOp);
}

#include "dnn/layers/gemm.hpp"
namespace chaos::inline dnn
{
	REGISTER_LAYER("GEMM", GEMM);
}

#include "dnn/layers/invert.hpp"
namespace chaos::inline dnn
{
	REGISTER_LAYER("Invert", Invert);
}

#include "dnn/layers/noop.hpp"
namespace chaos::inline dnn
{
	REGISTER_LAYER("Noop", Noop);
}

#include "dnn/layers/permute.hpp"
namespace chaos::inline dnn
{
	REGISTER_LAYER("Permute", Permute);
}

#include "dnn/layers/svd.hpp"
namespace chaos::inline dnn
{
	REGISTER_LAYER("SVD", SVD);
}