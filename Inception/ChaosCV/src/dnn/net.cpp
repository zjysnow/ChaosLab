#include "dnn/net.hpp"

namespace chaos
{
	Net::~Net()
	{
		if (opt.use_vulkan_compute)
		{
			for (auto& layer : layers) layer->DestroyPipeline(opt);
			if (opt.blob_vkallocator) delete opt.blob_vkallocator;
			if (opt.staging_vkallocator) delete opt.staging_vkallocator;
		}
	}

	void Net::Sequential()
	{
		for (int idx = 0; const auto& node : flow)
		{
			int producer = node.producer;
			if (producer != -1)
			{
				layers[producer]->tops_index.push_back(idx);
			}
			
			for (const auto& consumer : node.consumers)
			{
				layers[consumer]->bottoms_index.push_back(idx);
			}
			idx++;
		}

		if (opt.use_vulkan_compute)
		{
			for (auto& layer : layers) layer->CreatePipeline(opt);
		}
	}

	Extractor Net::CreateExtractor() const
	{
		return Extractor(this);
	}

	void Net::SetVulkanDevice(int device_index)
	{
		vkdev = GetGPUDevice(device_index);
		for (auto& layer : layers)
		{
			if (layer->support_vulkan) layer->vkdev = vkdev;
		}

		opt.use_vulkan_compute = true;
		opt.blob_vkallocator = new VulkanLocalAllocator(vkdev);
		opt.staging_vkallocator = new VulkanStagingAllocator(vkdev);
	}

	void Net::ForwardTo(int layer_index, std::vector<Tensor>& blobs, const Option& opt) const
	{
		auto layer = layers[layer_index];

		std::vector<Tensor> bottom_blobs(layer->bottoms_index.size());
		for (int i = 0; const auto & index : layer->bottoms_index)
		{
			if (blobs[index].empty())
			{
				ForwardTo(flow[index].producer, blobs, opt);
			}
			bottom_blobs[i++] = blobs[index];
		}

		if (layer->support_inplace)
		{
			layer->Forward(bottom_blobs, opt);
			for (int i = 0; const auto & index : layer->tops_index)
			{
				blobs[index] = bottom_blobs[i++];
			}
		}
		else
		{
			std::vector<Tensor> top_blobs(layer->tops_index.size());
			layer->Forward(bottom_blobs, top_blobs, opt);
			for (int i = 0; const auto & index : layer->tops_index)
			{
				blobs[index] = top_blobs[i++];
			}
		}
	}

	void Net::ForwardTo(int layer_index, std::vector<Tensor>& blobs, std::vector<VulkanTensor>& gpu_blobs, ComputeCommand& cmd, const Option& opt) const
	{
		auto layer = layers[layer_index];

		if (layer->support_vulkan)
		{
			std::vector<VulkanTensor> bottom_blobs(layer->bottoms_index.size());
			for (int i = 0; const auto & index : layer->bottoms_index)
			{
				if (gpu_blobs[index].empty() && blobs[index].empty())
				{
					ForwardTo(flow[index].producer, blobs, gpu_blobs, cmd, opt);
				}

				if (gpu_blobs[index].empty())
				{
					cmd.RecordUpload(blobs[index], gpu_blobs[index], opt);
				}

				bottom_blobs[i++] = gpu_blobs[index];
			}

			if (layer->support_inplace)
			{
				layer->Forward(bottom_blobs, cmd, opt);
				for (int i = 0; const auto & index : layer->tops_index)
				{
					gpu_blobs[index] = bottom_blobs[i++];
				}
			}
			else
			{
				std::vector<VulkanTensor> top_blobs(layer->tops_index.size());
				layer->Forward(bottom_blobs, top_blobs, cmd, opt);
				for (int i = 0; const auto & index : layer->tops_index)
				{
					gpu_blobs[index] = top_blobs[i++];
				}
			}
		}
		else
		{
			std::vector<Tensor> bottom_blobs(layer->bottoms_index.size());
			for (int i = 0; const auto & index : layer->bottoms_index)
			{
				if (blobs[index].empty() && gpu_blobs[index].empty())
				{
					ForwardTo(flow[index].producer, blobs, gpu_blobs, cmd, opt);
				}

				if (blobs[index].empty())
				{
					cmd.RecordDownload(gpu_blobs[index], blobs[index], opt);
					cmd.SubmitAndWait();
					cmd.Reset();
				}

				bottom_blobs[i++] = blobs[index];
			}

			if (layer->support_inplace)
			{
				layer->Forward(bottom_blobs, opt);
				for (int i = 0; const auto & index : layer->tops_index)
				{
					blobs[index] = bottom_blobs[i++];
				}
			}
			else
			{
				std::vector<Tensor> top_blobs(layer->tops_index.size());
				layer->Forward(bottom_blobs, top_blobs, opt);
				for (int i = 0; const auto & index : layer->tops_index)
				{
					blobs[index] = top_blobs[i++];
				}
			}
		}
	}

	Extractor::Extractor(const Net* net) : net(net)
	{
		blobs.resize(net->flow.size());
		gpu_blobs.resize(net->flow.size());
		opt = net->opt;
	}

	void Extractor::SetBlobData(const std::string& name, const Tensor& data)
	{
		int idx = net->flow.Find(name);
		blobs[idx] = data;
	}
	void Extractor::GetBlobData(const std::string& name, Tensor& data)
	{
		int idx = net->flow.Find(name);
		if (blobs[idx].empty())
		{
			if (opt.use_vulkan_compute)
			{
				ComputeCommand cmd(net->vkdev);

				int layer_index = net->flow[idx].producer;
				net->ForwardTo(layer_index, blobs, gpu_blobs, cmd, opt);
				
				if (blobs[idx].empty() && not gpu_blobs[idx].empty())
				{
					cmd.RecordDownload(gpu_blobs[idx], blobs[idx], opt);
				}
				cmd.SubmitAndWait();
				cmd.Reset();
			}
			else
			{
				int layer_index = net->flow[idx].producer;
				net->ForwardTo(layer_index, blobs, opt);
			}
		}
		data = blobs[idx];
	}

	void Extractor::SetVulkanCompute(bool enable)
	{
		if (net->opt.use_vulkan_compute)
		{
			opt.use_vulkan_compute = enable;
		}
		else
		{
			LOG(WARNING) << "can not set to vulkan compute, use cpu";
		}
	}
}