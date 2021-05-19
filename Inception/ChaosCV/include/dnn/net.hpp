#pragma once

#include "core/core.hpp"
#include "dnn/blob.hpp"
#include "dnn/layer.hpp"

namespace chaos
{
	class Extractor;
	class CHAOS_API Net
	{
	public:
		virtual ~Net() = default;
		Extractor CreateExtractor() const;
		
		template<class...Layers>
		static Net Sequential(Layers... layers)
		{
			Net net;
			net.layers = { Cast(layers)... };

			size_t blob_count = 0;
			for (const auto& layer : net.layers)
			{
				blob_count += layer->bottoms_count();
				blob_count += layer->tops_count();
			}
			net.flows.reserve(blob_count * 10);
			return net;
		}

		template<int idx, class...Blobs>
		Blob layer(Blobs...blobs)
		{
			auto GetNames = [](const Blob& blob) { return blob.name; };
			std::vector<std::string> names = { GetNames(blobs)... };
			for (const auto& name : names)
			{
				int i = blob_index(name);
				flows[i].consumers.push_back(idx);
				layers[idx]->bottoms_idx.push_back(i);
			}
			Blob blob;
			blob.producer = idx;
			blob.net = this;
			return blob;
		}

		Blob blob(const std::string& name)
		{
			for (auto& blob : flows)
			{
				if (blob.name == name) return blob;
			}
			Blob blob(name);
			blob.net = this;
			flows.push_back(blob);
			return blob;
		}

	public:
		template<class Type>
		static Ptr<Layer> Cast(const Type& layer)
		{
			return std::make_shared<Type>(layer);
		}

		void ForwardTo(int layer_index, std::vector<Tensor>& blobs, const Option& opt) const;

		int blob_index(const std::string& name) const
		{
			for (int idx = 0; const auto & blob : flows)
			{
				if (blob.name == name) return idx;
				idx++;
			}
			LOG(FATAL);
			return -1;
		}

		friend class Extractor;
		
		std::vector<Ptr<Layer>> layers;
		std::vector<Blob> flows;
	};

	class CHAOS_API Extractor
	{
	public:
		virtual ~Extractor() = default;

		void SetBlobData(const std::string& name, const Tensor& data);
		void GetBlobData(const std::string& name, Tensor& data);

	protected:
		Extractor(const Net* net, size_t blobs_cnt);

		friend Extractor Net::CreateExtractor() const;
		const Net* net;
		Option opt;
		std::vector<Tensor> blobs;
	};
}