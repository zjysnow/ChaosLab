#include "dnn/layer.hpp"
#include "dnn/option.hpp"
#include "dnn/layer_factory.hpp"

#include "dnn/layers/abs.hpp"
#include "dnn/layers/vulkan/abs_vulkan.hpp"
namespace chaos
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