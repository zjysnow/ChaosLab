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
			POW,
		};

		BinaryOp();
		BinaryOp(const std::string& name, int type);

		void Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const override;
		void Set(const std::string& pname, const std::any& val) override;

		size_t bottoms_count() const override { return 2; }
		size_t tops_count() const override { return 1; };

		int op_type = ADD;
	};
}