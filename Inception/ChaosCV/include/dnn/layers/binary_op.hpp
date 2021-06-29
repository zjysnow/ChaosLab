#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	inline namespace dnn
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

			void Set(const std::string& pname, const std::any& param) override;

			int op_type = ADD;
		};
	}
}