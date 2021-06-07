#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	class CHAOS_API BinaryOp : public Layer
	{
	public:
		enum OpType
		{
			ADD,
			SUB,
			MUL,
			DIV,
		};

		BinaryOp();
		BinaryOp(int op_type);

		void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const override;

		int op_type = ADD;
	};
}