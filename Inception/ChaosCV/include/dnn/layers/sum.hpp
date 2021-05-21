#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	namespace dnn
	{
		class CHAOS_API Sum : public Layer
		{
		public:
			Sum();

			void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const override;

			void Set(const std::string& pname, const std::any& val) final;

			bool all = false;
			std::vector<uint32> vecdim = { 0 };
		};
	}
}