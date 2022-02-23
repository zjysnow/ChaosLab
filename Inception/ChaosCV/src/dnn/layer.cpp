#include "dnn/layer.hpp"

namespace chaos
{
	namespace dnn
	{
		Layer::Layer(const std::string& type) : type(type) {}

		void Layer::Forward(const Tensor& input_blob, Tensor& output_blob, const Option& opt) const
		{
			CHECK(support_inplace);

			output_blob = input_blob.Clone(opt.blob_allocator);
			CHECK(not output_blob.empty());

			Forward(output_blob, opt);
		}

		void Layer::Forward(Tensor& input_output_blob, const Option& opt) const
		{
			LOG(FATAL);
		}

		void Layer::Forward(const std::vector<Tensor>& input_blobs, std::vector<Tensor>& output_blobs, const Option& opt) const
		{
			CHECK(support_inplace);

			output_blobs.resize(input_blobs.size());
			for (int i = 0; i < input_blobs.size(); i++)
			{
				output_blobs[i] = input_blobs[i].Clone(opt.blob_allocator);
				CHECK(not output_blobs[i].empty());
			}

			Forward(output_blobs, opt);
		}

		void Layer::Forward(std::vector<Tensor>& blobs, const Option& opt) const
		{
			LOG(FATAL);
		}
	}
}