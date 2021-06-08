#pragma once

#include "core/core.hpp"
#include "core/tensor.hpp"
#include "core/command.hpp"
#include "core/pipeline.hpp"

#include "dnn/option.hpp"

#include <any>
#include <cmath>
#include <algorithm>

namespace chaos
{
	class CHAOS_API Layer
	{
	public:
		Layer(const std::string& type);

		virtual void CreatePipeline(const Option& opt);
		virtual void DestroyPipeline(const Option& opt);

		virtual void Forward(std::vector<Tensor>& bottom_top_blobs, const Option& opt = Option()) const;
		virtual void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt = Option()) const;

		virtual void Forward(std::vector<VulkanTensor>& bottom_top_blobs, ComputeCommand& cmd, const Option& opt = Option()) const;
		virtual void Forward(const std::vector<VulkanTensor>& bottom_blobs, std::vector<VulkanTensor>& top_blobs, ComputeCommand& cmd, const Option& opt = Option()) const;

		virtual void Set(const std::string& pname, const std::any& param);

		const std::string& type;

		// support inplace inference
		bool support_inplace;
		bool support_vulkan;

		const VulkanDevice* vkdev;
	};
}