#include "dnn/layers/std.hpp"
#include "utils/op.hpp"

namespace chaos
{
	STD::STD() : Layer("STD") {}

	void STD::Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const
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
		//Tensor mean, diff;
		//Operator::Mean(bottom_blob, dim, mean);
		//Operator::Sub(bottom_blob, mean, diff);

		Tensor s, m = mean(bottom_blob, dim);
	}
}