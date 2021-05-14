#include "dnn/layers/mean.hpp"
#include "utils/op.hpp"

namespace chaos
{
	Mean::Mean() : Layer("Mean") {}

	void Mean::Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const
	{
		CHECK_LT(dim, (int)bottom_blob.shape.dims);

		size_t n = 0;
		
		if (bottom_blob.shape.dims == 1 or dim == -1)
		{
			if (top_blob.empty()) top_blob.Create({ 1u }, { 1u }, DataType::D4, Packing::CHW, opt.blob_allocator);
			CHECK_EQ(Shape(1u), top_blob.shape);
			n = bottom_blob.shape.total(); // bottom_blob.shape[0];
		}
		else
		{
			uint32 d = bottom_blob.shape[dim];
			if (top_blob.empty()) top_blob.Create({ d }, { 1u }, DataType::D4, Packing::CHW, opt.blob_allocator);
			CHECK_EQ(Shape(d), top_blob.shape);
			n = bottom_blob.shape.total() / d;
		}
		// m = sum(a) / n;
		Operator::Sum(bottom_blob, dim, top_blob);
		Operator::Mul(1.f / n, top_blob, top_blob);
	}

	void Mean::Set(const std::string& pname, const std::any& val)
	{
		if ("dim" == pname)
		{
			dim = std::any_cast<int>(val);
			CHECK_GE(dim, -1) << "dim should greater than -1";
		}
	}
}