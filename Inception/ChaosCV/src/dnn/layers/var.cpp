#include "dnn/layers/var.hpp"
#include "dnn/layers/reduce.hpp"
#include "dnn/layers/binary_op.hpp"

namespace chaos
{
	inline namespace dnn
	{
		Var::Var() : Layer("Var") 
		{
			mean = std::make_shared<Reduce>();
			mean->Set("op_type", Reduce::AVG);

			sub2 = std::make_shared<BinaryOp>();
			sub2->Set("op_type", BinaryOp::SUB2);
		}

		void Var::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const
		{
			CHECK_EQ(1, bottom_blobs.size()) << "layer Sum expect 1 input but got " << bottom_blobs.size();
			CHECK_EQ(1, top_blobs.size()) << "layer Sum expect 1 output but got " << top_blobs.size();

			const Tensor& A = bottom_blobs[0];
			int dims = (int)A.shape.size();
			CHECK_LE(vecdim.size(), dims) << "dims of A should greater-equal than size of vecdims";

			Shape shape = A.shape;
			float N = 1.f;
			if (all)
			{
				for (int i = 0; i < dims; i++)
				{
					N *= shape[i];
					shape[i] = 1;
				}
			}
			else
			{
				for (const auto& i : vecdim)
				{
					CHECK_LE(i, dims) << "out of range";
					N *= shape[i];
					shape[i] = 1;
				}
			}

			Tensor& V = top_blobs[0];
			if (V.empty()) V.Create(shape, shape.steps(), Depth::D4, Packing::CHW, opt.blob_allocator);
			CHECK_EQ(shape, V.shape);

			mean->Set("all", all);
			mean->Set("vecdim", vecdim);
			mean->Set("alpha", 1.f);
			mean->Forward(bottom_blobs, top_blobs, opt);

			std::vector<Tensor> S(1);
			sub2->Forward({ A,V }, S, opt);

			mean->Set("alpha", unbias ? N / (N - 1) : 1.f);
			mean->Forward(S, top_blobs, opt);
		}

		void Var::Set(const std::string& pname, const std::any& param)
		{
			if ("unbias" == pname) unbias = std::any_cast<bool>(param);
			if ("all" == pname) all = std::any_cast<bool>(param);
			if ("vecdim" == pname) vecdim = std::any_cast<Array<int>>(param);
		}
	}
}