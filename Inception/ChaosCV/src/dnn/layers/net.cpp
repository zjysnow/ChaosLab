#include "dnn/net.hpp"

namespace chaos
{
	namespace dnn
	{
		void Net::ForwardTo(int layer_index, std::vector<Tensor>& blobs, const Option& opt) const
		{
			//auto layer = layers[layer_index];

			//std::vector<Tensor> bottoms(layer->bottoms_count());
			//for (size_t i = 0; i < bottoms.size(); i++)
			//{
			//	size_t index = layer->bottoms_idx[i];
			//	if (blobs[index].empty())
			//	{
			//		
			//		ForwardTo(flows[index].producer, blobs, opt);
			//	}
			//	bottoms[i] = blobs[index];
			//}
			//std::vector<Tensor> tops(layer->tops_count());
			//layer->Forward(bottoms, tops, opt);
			//for (size_t i = 0; i < tops.size(); i++)
			//{
			//	int index = layer->tops_idx[i];
			//	blobs[index] = tops[i];
			//}
		}

		Extractor Net::CreateExtractor() const
		{
			return Extractor(this, flows.size());
		}

		Extractor::Extractor(const Net* net, size_t blobs_cnt) : net(net)
		{
			blobs.resize(blobs_cnt);
		}

		void Extractor::SetBlobData(const std::string& name, const Tensor& data)
		{
			int index = net->blob_index(name);
			blobs[index] = data;
		}
		void Extractor::GetBlobData(const std::string& name, Tensor& data)
		{
			int index = net->blob_index(name);
			if (blobs[index].empty())
			{
				int layer_index = net->flows[index].producer;
				net->ForwardTo(layer_index, blobs, opt);
			}
			data = blobs[index];
		}
	}
}