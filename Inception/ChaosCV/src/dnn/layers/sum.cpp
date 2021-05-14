#include "dnn/layers/sum.hpp"

namespace chaos
{
	Sum::Sum() : Layer("Sum") {}

	void Sum::Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const
	{
		CHECK_LT(dim, (int)bottom_blob.shape.dims);
		size_t dims = bottom_blob.shape.dims;
		if (dims == 1) // always continuous
		{
			if (top_blob.empty()) top_blob.Create({ 1u }, { 1u }, DataType::D4, Packing::CHW, opt.blob_allocator);
			CHECK_EQ(Shape(1u), top_blob.shape);

			uint32 n = bottom_blob.shape[0];
			float sum = 0.f;
			for (uint32 i = 0; i < n; i++)
			{
				sum += bottom_blob[i];
			}
			top_blob[0] = sum;
		}
		else // bottom_blob.shape.dims >= 2
		{
			if (dim == -1)
			{
				if (top_blob.empty()) top_blob.Create({ 1u }, { 1u }, DataType::D4, Packing::CHW, opt.blob_allocator);
				CHECK_EQ(Shape(1u), top_blob.shape);
				uint32 n = bottom_blob.total();
				float sum = 0.f;
				if (bottom_blob.is_continuous())
				{
					for (uint32 i = 0; i < n; i++)
					{
						sum += bottom_blob[i];
					}
					top_blob[0] = sum;
				}
				else
				{
					for (size_t i = 0; i < n; i++)
					{
						size_t a_idx = 0;
						size_t idx = i;
						for (size_t d = 0; d < dims; d++)
						{
							size_t k = idx % bottom_blob.shape[dims - d - 1];
							a_idx += k * bottom_blob.steps[dims - d - 1];
							idx /= bottom_blob.shape[dims - d - 1];
						}
						sum += bottom_blob[a_idx];
					}
				}
			}
			else
			{
				uint32 d = bottom_blob.shape[dim];
				Shape shape = bottom_blob.shape;
				for (size_t i = 0; i < dims; i++) // reset other dim to 1
				{
					if (i != dim) shape[i] = 1;
				}
				Steps steps = shape.steps();

				if (top_blob.empty()) top_blob.Create({ d }, { 1u }, DataType::D4, Packing::CHW, opt.blob_allocator);
				CHECK_EQ(Shape(d), top_blob.shape);

				memset(top_blob.data, 0, d * sizeof(float));

				size_t n = bottom_blob.shape.total();
				for (size_t i = 0; i < n; i++)
				{
					size_t a_idx = 0;
					size_t c_idx = 0;
					size_t idx = i;
					for (size_t d = 0; d < dims; d++)
					{
						size_t k = idx % bottom_blob.shape[dims - d - 1];
						a_idx += (k >= shape[dims - d - 1] ? 0 : k) * steps[dims - d - 1];
						c_idx += k * bottom_blob.steps[dims - d - 1];
						idx /= bottom_blob.shape[dims - d - 1];
					}
					top_blob[a_idx] += bottom_blob[c_idx];
				}
			}
		}
	}

	void Sum::Set(const std::string& pname, const std::any& val)
	{
		if ("dim" == pname)
		{
			dim = std::any_cast<int>(val);
		}
	}
}