#include "dnn/layers/std.hpp"
#include "utils/op.hpp"

namespace chaos
{
	STD::STD() : Layer("STD") {}

	void STD::Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const
	{
		CHECK_LT(dim, bottom_blob.shape.dims);
		if (bottom_blob.shape.dims == 1) // always continuous
		{
			if (top_blob.empty()) top_blob.Create({ 1u }, { 1u }, DataType::D4, Packing::CHW, opt.blob_allocator);
			CHECK_EQ(Shape(1u), top_blob.shape);

			uint32 n = bottom_blob.shape[0];
			Tensor mean;
			Operator::Mean(bottom_blob, 0, mean);
			float sigma = 0.f;
			for (uint32 i = 0; i < n; i++)
			{
				sigma += std::pow(bottom_blob[i] - mean[0], 2.f);
			}
			sigma = std::sqrt(sigma / (n - 1));
			top_blob[0] = sigma;
		}
		else
		{

		}
	}
}