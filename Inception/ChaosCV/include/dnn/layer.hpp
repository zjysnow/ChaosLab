#pragma once

#include "core/tensor.hpp"
#include "dnn/option.hpp"

namespace chaos
{
	namespace dnn
	{
		class CHAOS_API Layer
		{
		public:
			Layer(const std::string& type);

			virtual ~Layer() = default;

			virtual void Forward(const Tensor& input_blob, Tensor& output_blob, const Option& opt = Option()) const;
			virtual void Forward(Tensor& input_output_blob, const Option& opt = Option()) const;

			virtual void Forward(const std::vector<Tensor>& input_blobs, std::vector<Tensor>& output_blobs, const Option& opt = Option()) const;
			virtual void Forward(std::vector<Tensor>& input_output_blobs, const Option& opt = Option()) const;

			const std::string type;

			bool support_inplace = false;
		};
	}

	
}