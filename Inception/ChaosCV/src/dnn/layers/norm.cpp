#include "dnn/layers/norm.hpp"
#include <functional>

namespace chaos
{
	Norm::Norm() : Layer("Norm") {}

	void Norm::Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const
	{
		CHECK_LE(1, bottom_blob.shape.dims);
		CHECK_GE(2, bottom_blob.shape.dims) << "input should be a vector or a matrix";

		if (top_blob.empty()) top_blob.Create({ 1u }, { 1u }, DataType::D4, Packing::CHW, opt.blob_allocator);
		CHECK_EQ(Shape(1u), top_blob.shape);

		if (1 == bottom_blob.shape.dims)
		{
			if (isfinite(p))
			{
				float norm = 0.f;
				for (size_t i = 0; i < bottom_blob.shape[0]; i++)
				{
					norm += std::powl(std::abs(bottom_blob[i]), p);
				}
				top_blob[0] = std::powl(norm, 1.f / p);
			}
			else
			{
				if (std::signbit(p)) // p == -INFINITY
				{
					float norm = FLT_MAX;
					for (size_t i = 0; i < bottom_blob.shape[0]; i++)
					{
						norm = std::min(norm, std::abs(bottom_blob[i]));
					}
					top_blob[0] = norm;
				}
				else // p == INFINITY
				{
					float norm = -FLT_MAX;
					for (size_t i = 0; i < bottom_blob.shape[0]; i++)
					{
						norm = std::max(norm, std::abs(bottom_blob[i]));
					}
					top_blob[0] = norm;
				}
			}
		}
		else // 2 == bottom_blob.shape.dims
		{
			if (std::abs(p - 2) < FLT_EPSILON) // p == 2
			{

			}
			if (std::abs(p - 1) < FLT_EPSILON) // p == 1
			{

			}
			if (not std::signbit(p)) // p == INFINITY
			{

			}
		}
	}

	void Norm::Set(const std::string& pname, const std::any& val)
	{
		if ("p" == pname)
		{
			p = std::any_cast<float>(val);
		}
	}
}