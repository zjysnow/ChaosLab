#include "dnn/layers/var.hpp"
#include "dnn/layers/mean.hpp"
#include "dnn/layers/binary_op.hpp"

namespace chaos
{
	inline namespace dnn
	{
		Var::Var() : Layer("Var") 
		{
			mean = std::make_shared<Mean>();
			sub2 = std::make_shared<BinaryOp>();
			sub2->Set("op_type", BinaryOp::SUB2);
		}

		void Var::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const
		{
			CHECK_EQ(1, bottom_blobs.size()) << "layer Sum expect 1 input but got " << bottom_blobs.size();
			CHECK_EQ(1, top_blobs.size()) << "layer Sum expect 1 output but got " << top_blobs.size();

			std::vector<Tensor> M(1), N(1);

			mean->Set("all", all);
			mean->Set("vecdim", vecdim);

			mean->Set("unbias", false);
			mean->Forward(bottom_blobs, M, opt);

			sub2->Forward({bottom_blobs[0], M[0]}, N, opt);
			
			mean->Set("unbias", unbias);
			mean->Forward(N, top_blobs, opt);
		}

		void Var::Set(const std::string& pname, const std::any& param)
		{
			if ("unbias" == pname) unbias = std::any_cast<bool>(param);
			if ("all" == pname) all = std::any_cast<bool>(param);
			if ("vecdim" == pname) vecdim = std::any_cast<Array<int>>(param);
		}
	}
}