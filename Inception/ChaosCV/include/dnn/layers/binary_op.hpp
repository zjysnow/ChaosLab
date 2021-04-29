#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	// support broadcast
	class CHAOS_API BinaryOp : public Layer
	{
	public:
		enum OpType
		{
			ADD,
			SUB,
			MUL,
			DIV,
			MAX,
			MIN,
		};

		BinaryOp();

		void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const override;
		void Set(const std::string& pname, const std::any& val) override;

		int op_type = ADD;
	};
}