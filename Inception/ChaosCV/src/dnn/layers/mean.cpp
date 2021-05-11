#include "dnn/layers/mean.hpp"
#include "utils/op.hpp"
namespace chaos
{
	Mean::Mean() : Layer("Mean") {}

	void Mean::Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const
	{
		CHECK_LT(dim, bottom_blob.shape.dims);
		if (bottom_blob.shape.dims == 1) // always continuous
		{
			if (top_blob.empty()) top_blob.Create({ 1u }, { 1u }, DataType::D4, Packing::CHW, opt.blob_allocator);
			CHECK_EQ(Shape(1u), top_blob.shape);

			uint32 n = bottom_blob.shape[0];
			Tensor sum;
			Operator::Sum(bottom_blob, 0, sum);
			top_blob[0] = sum[0] / (float)n;
		}
		else // bottom_blob.shape.dims >= 2
		{
			LOG(FATAL) << "not now";
			if (-1 == dim)
			{
				if (top_blob.empty()) top_blob.Create({ 1u }, { 1u }, DataType::D4, Packing::CHW, opt.blob_allocator);
				CHECK_EQ(Shape(1u), top_blob.shape);
			}
			else // dim != -1
			{
				uint32 d = bottom_blob.shape[dim];
				if (top_blob.empty()) top_blob.Create({ d }, { 1u }, DataType::D4, Packing::CHW, opt.blob_allocator);
				CHECK_EQ(Shape(d), top_blob.shape);
			}
		}
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