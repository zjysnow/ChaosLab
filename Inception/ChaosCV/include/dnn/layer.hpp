#pragma once

#include "core/core.hpp"
#include "core/tensor.hpp"

#include "dnn/option.hpp"

namespace chaos
{
	class CHAOS_API Layer
	{
	public:
		Layer(const std::wstring& type);

		//virtual void Forward(const Tensor& bottom, Tensor& top, const Option& opt = Option()) const;
		//virtual void Forward(Tensor& bottom_top, const Option& opt = Option()) const;

		virtual void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt = Option()) const;
		virtual void Forward(std::vector<Tensor>& bottom_top_blobs, const Option& opt = Option()) const;

		const std::wstring type;

		bool support_inplace;
		bool support_vulkan;
	};
}