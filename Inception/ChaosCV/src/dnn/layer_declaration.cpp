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

//#include "dnn/layers/back_subst.hpp"
//namespace chaos::inline dnn
//{
//	REGISTER_LAYER("BackSubst", BackSubst);
//}
//
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
//
//#include "dnn/layers/gemm.hpp"
//namespace chaos::inline dnn
//{
//	REGISTER_LAYER("GEMM", GEMM);
//}
//
//#include "dnn/layers/invert.hpp"
//namespace chaos::inline dnn
//{
//	REGISTER_LAYER("Invert", Invert);
//}
//
//#include "dnn/layers/permute.hpp"
//namespace chaos::inline dnn
//{
//	REGISTER_LAYER("Permute", Permute);
//}
//
#include "dnn/layers/sum.hpp"
#include "dnn/layers/vulkan/sum_vulkan.hpp"
namespace chaos::inline dnn
{
	class SumFinal : public SumVulkan
	{
	public:
		void CreatePipeline(const Option& opt) final
		{
			Sum::CreatePipeline(opt);
			if (opt.use_vulkan_compute) SumVulkan::CreatePipeline(opt);
		}
		void DestroyPipeline(const Option& opt) final
		{
			if (opt.use_vulkan_compute) SumVulkan::DestroyPipeline(opt);
			Sum::DestroyPipeline(opt);
		}
	};
	REGISTER_LAYER("Sum", SumFinal);
}

//#include "dnn/layers/svd.hpp"
//namespace chaos::inline dnn
//{
//	REGISTER_LAYER("SVD", SVD);
//}