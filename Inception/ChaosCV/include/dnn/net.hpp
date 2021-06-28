#pragma once

#include "core/core.hpp"

#include "dnn/layer.hpp"
#include "dnn/flow.hpp"

namespace chaos
{
	class Extractor;
	class CHAOS_API Net
	{
	public:
		Net() = default;
		~Net();

		template<class...Args>
		Net(const Args&...args)
		{
			layers = { CreateLayer(args)... };
		}
		template<class...Args>
		Flow& blob(const Args&...args)
		{
			std::vector<std::string> names = { args... };
			for (const auto& name : names)
			{
				flow.AddNode(name);
			}
			return flow;
		}
		template<int idx, class...Args>
		int layer(const Args&...args)
		{
			std::vector<std::string> names = { args... };
			for (const auto& name : names)
			{
				auto& node = flow.AddNode(name);
				node.consumers.push_back(idx);
			}
			return idx;
		}
		void Sequential();

		Extractor CreateExtractor() const;

		void SetVulkanDevice(int device_index);
	protected:
		template<class Type>
		Ptr<Layer> CreateLayer(const Type& args)
		{
			return std::make_shared<Type>(args);
		}

		friend class Extractor;
		void ForwardTo(int layer_index, std::vector<Tensor>& blobs, const Option& opt) const;
		void ForwardTo(int layer_index, std::vector<Tensor>& blobs, std::vector<VulkanTensor>& gpu_blobs, ComputeCommand& cmd, const Option& opt) const;

		std::vector<Ptr<Layer>> layers;
		Flow flow;
		Option opt;
		const VulkanDevice* vkdev;
	};

	class CHAOS_API Extractor
	{
	public:
		void SetBlobData(const std::string& name, const Tensor& data);
		void GetBlobData(const std::string& name, Tensor& data);

		void SetVulkanCompute(bool enable);
	protected:
		friend Extractor Net::CreateExtractor() const;
		Extractor(const Net* net);

		const Net* net;
		Option opt;
		std::vector<Tensor> blobs;
		std::vector<VulkanTensor> gpu_blobs;
	};


}