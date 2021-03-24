#include "dnn/layer.hpp"

namespace chaos
{
	Layer::Layer(const std::wstring& type) : type(type) 
	{
		support_inplace = false;
		support_vulkan = false;
	}

	void Layer::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const
	{
		CHECK(support_inplace) << "layer '" << type << "' do not be implemented";

		top_blobs = bottom_blobs;
		for (size_t i = 0; i < top_blobs.size(); i++)
		{
			top_blobs[i] = bottom_blobs[i].Clone(opt.blob_allocator);
			CHECK(not top_blobs[i].empty());
		}
		Forward(top_blobs, opt);
	}
	void Layer::Forward(std::vector<Tensor>& bottom_top_blobs, const Option& opt) const
	{
		LOG(FATAL) << "layer " << type << " do not support inpalce";
	}
}