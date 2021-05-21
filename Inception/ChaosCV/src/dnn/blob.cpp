#include "dnn/blob.hpp"
#include "dnn/net.hpp"

namespace chaos
{
	namespace dnn
	{
		Blob& Blob::operator=(const Blob& blob)
		{
			producer = blob.producer;

			//int idx = net->blob_index(name);
			//net->flows[idx].producer = blob.producer;
			//net->layers[producer]->tops_idx.push_back(idx);

			return *this;
		}
	}
}