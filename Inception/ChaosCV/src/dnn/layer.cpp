#include "dnn/layer.hpp"

namespace chaos
{
	namespace dnn
	{
		Layer::Layer(const std::string& type) : type(type)
		{
			support_inplace = false;
			support_vulkan = false;
		}

		//void Layer::Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const
		//{
		//	LOG(FATAL);
		//}
		//void Layer::Forward(Tensor& bottom_top_blob, const Option& opt) const
		//{
		//	LOG(FATAL);
		//}

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
		void Layer::Forward(std::vector<Tensor>&, const Option&) const
		{
			LOG(FATAL);
		}

		void Layer::Set(const std::string&, const std::any&) {}
	}
}