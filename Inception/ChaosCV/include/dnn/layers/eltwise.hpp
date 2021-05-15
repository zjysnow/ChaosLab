#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	class Eltwise : public Layer
	{
	public:
		enum OpType
		{
			SUM,
			PROD,
		};

		Eltwise();

		void Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const override;
		void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const override;

		void Set(const std::string& pname, const std::any& val) override;

		int op_type
	};
}