#include "dnn/layer.hpp"
#include "dnn/option.hpp"

namespace chaos
{
	Layer::Layer(const std::string& type) : type(type)
	{
		support_inplace = false;
		support_vulkan = false;
	}

	void Layer::CreatePipeline(const Option&) {}
	void Layer::DestroyPipeline(const Option&) {}

	void Layer::Forward(std::vector<Tensor>&, const Option&) const
	{
		LOG(FATAL);
	}
	void Layer::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const
	{
		CHECK(support_inplace);
		top_blobs.resize(bottom_blobs.size());
		for (size_t i = 0; i < bottom_blobs.size(); i++)
		{
			top_blobs[i] = bottom_blobs[i].Clone(opt.blob_allocator);
		}
		Forward(top_blobs, opt);
	}

	void Layer::Forward(std::vector<VulkanTensor>&, ComputeCommand&, const Option&) const
	{
		LOG(FATAL);
	}
	void Layer::Forward(const std::vector<VulkanTensor>& bottom_blobs, std::vector<VulkanTensor>& top_blobs, ComputeCommand& cmd, const Option& opt) const
	{
		CHECK(support_inplace);
		top_blobs.resize(bottom_blobs.size());
		for (size_t i = 0; i < bottom_blobs.size(); i++)
		{
			cmd.RecordClone(bottom_blobs[i], top_blobs[i], opt);
		}
		Forward(top_blobs, cmd, opt);
	}
}