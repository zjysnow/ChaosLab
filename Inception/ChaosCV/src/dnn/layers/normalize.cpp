#include "dnn/layers/normalize.hpp"

#include "utils/op.hpp"

namespace chaos
{
	Normalize::Normalize() : Layer("Normalize") {}



	void Normalize::Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const
	{
		const Shape& shape = bottom_blob.shape;
		size_t dims = shape.dims;
		
		if (top_blob.empty()) top_blob.Create(shape, shape.steps(), DataType::D4, Packing::CHW, opt.blob_allocator);
		CHECK_EQ(shape, top_blob.shape);

		if (dims == 1)
		{
			if (ZSCORE == method)
			{
				float m = mean(bottom_blob);
			}
			if (NORM == method)
			{
				float n = norm(bottom_blob, f1);
				for (size_t i = 0; i < shape[0]; i++)
				{
					top_blob[i] = bottom_blob[i] / n;
				}
			}
			if (RANGE == method)
			{

			}
		}
		else // dims >= 2
		{
			
		}
	}

	void Normalize::Set(const std::string& pname, const std::any& val)
	{
		if ("method" == pname)
		{
			method = std::any_cast<Method>(val);
			switch (method) // to set default flag
			{
			case NORM:
				f1 = 2.f;
				break;
			case RANGE:
				f1 = 0.f;
				f2 = 1.f;
			default:
				break;
			}
		}
		if ("f1" == pname)
		{
			f1 = std::any_cast<float>(val);
		}
		if ("f2" == pname)
		{
			f2 = std::any_cast<float>(val);
		}
		if ("dim" == pname)
		{
			dim = std::any_cast<int>(val);
		}
	}
}