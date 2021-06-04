#include "dnn/layer.hpp"

namespace chaos
{
	Layer::Layer(const std::string& type) : type(type)
	{
		support_inplace = false;
		support_vulkan = false;
	}

	void Layer::CreatePipeline() {}
	void Layer::DestroyPipeline() {}

	void Layer::Forward(std::vector<Tensor>& bottom_top_blobs) const
	{

	}
	void Layer::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs) const
	{

	}

	void Layer::Forward(std::vector<VulkanTensor>& bottom_top_blobs, ComputeCommand& cmd) const
	{

	}
	void Layer::Forward(const std::vector<VulkanTensor>& bottom_blobs, std::vector<VulkanTensor>& top_blobs, ComputeCommand& cmd) const
	{

	}
}