#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	class CHAOS_API Sum : public Layer
	{
	public:
		Sum();

		void Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const override;
		void Set(const std::string& pname, const std::any& val);


		bool all = false;
		std::vector<uint32> vecdim = { 0 };
	};
}