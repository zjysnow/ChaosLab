#include "dnn/layers/mean.hpp"
#include "dnn/layers/sum.hpp"
#include "dnn/layers/binary_op.hpp"

namespace chaos
{
	inline namespace dnn
	{
		Mean::Mean() : Layer("Mean")
		{
			sum = std::make_shared<Sum>();
			div = std::make_shared<BinaryOp>();
			div->Set("op_type", BinaryOp::DIV);
		}

		void Mean::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const
		{
			CHECK_EQ(1, bottom_blobs.size()) << "layer Sum expect 1 input but got " << bottom_blobs.size();
			CHECK_EQ(1, top_blobs.size()) << "layer Sum expect 1 output but got " << top_blobs.size();

			sum->Set("all", all);
			sum->Set("vecdim", vecdim);

			const Tensor& A = bottom_blobs[0];
			int dims = (int)A.shape.size();
			CHECK_LE(vecdim.size(), dims) << "dims of A should greater-equal than size of vecdims";

			const Shape& shape = A.shape;
			float N = 1.f;
			if (all)
			{
				for (int i = 0; i < dims; i++)
				{
					N *= shape[i];
				}
			}
			else
			{
				for (const auto& i : vecdim)
				{
					CHECK_LE(i, dims) << "out of range";
					N *= shape[i];
				}
			}
			
			sum->Forward(bottom_blobs, top_blobs, opt);
			div->Forward({ top_blobs[0], {N} }, top_blobs, opt);
		}

		void Mean::Set(const std::string& pname, const std::any& param)
		{
			if ("vecdim" == pname) vecdim = std::any_cast<Array<int>>(param);
		}
	}
}