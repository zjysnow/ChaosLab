#include "dnn/layer.hpp"

namespace chaos
{
	Layer::Layer(const std::string& type) : type(type) 
	{
		support_inplace = false;
		support_vulkan = false;
	}

	void Layer::Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const
	{
		LOG(FATAL);
	}
	void Layer::Forward(Tensor& bottom_top_blob, const Option& opt) const
	{
		LOG(FATAL);
	}

	void Layer::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const
	{
		LOG(FATAL);
	}
	void Layer::Forward(std::vector<Tensor>& bottom_top_blobs, const Option& opt) const
	{
		LOG(FATAL);
	}
}