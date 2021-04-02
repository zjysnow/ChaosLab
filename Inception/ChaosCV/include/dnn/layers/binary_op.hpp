#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	// support broadcast
	class CHAOS_API BinaryOp : public Layer
	{
	public:
		enum OperationType
		{
			ADD,
			SUB,
			MUL,
			DIV,
		};

		BinaryOp();

		void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const override;

		int op_type = ADD;
	};
}