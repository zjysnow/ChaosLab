#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	namespace dnn
	{
		class CHAOS_API Norm : public Layer
		{
		public:
			Norm();

			void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const override;
			void Set(const std::string& pname, const std::any& val) final;
			float p = 2.f;
		};
	}
}