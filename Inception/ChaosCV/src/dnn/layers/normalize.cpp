#include "dnn/layers/normalize.hpp"

namespace chaos
{
	Normalize::Normalize() : Layer("Normalize") {}

	void Normalize::Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const
	{
		//size_t total = bottom_blob.total();
		//CHECK_EQ(total, bottom_blob.shape.total());

		const Shape shape = bottom_blob.shape;
		const Steps steps = bottom_blob.steps;
		top_blob.Create(shape, shape.steps(), DataType::D4, Packing::CHW, opt.blob_allocator);

		auto valid_index = [=](size_t i) {
			if (bottom_blob.is_continuous())
			{
				return i;
			}
			else
			{
				size_t idx = 0;
				size_t dims = shape.dims;
				for (size_t d = 0; d < dims; d++)
				{
					size_t k = i % shape[dims - d - 1];
					idx += k * steps[dims - d - 1];
					i /= shape[dims - d - 1];
				}
				return idx;
			}
		};

		if (norm_type == L1)
		{
			float norm = 0.f;
			for (size_t i = 0; i < shape.total(); i++)
			{
				norm += bottom_blob[valid_index(i)];
			}
			for (size_t i = 0; i < shape.total(); i++)
			{
				top_blob[i] = bottom_blob[valid_index(i)] / norm;
			}
		}
		if (norm_type == L2)
		{
			float norm = 0.f;
			for (size_t i = 0; i < shape.total(); i++)
			{
				size_t idx = valid_index(i);
				norm += bottom_blob[idx] * bottom_blob[idx];
			}
			norm = std::sqrt(norm);

			for (size_t i = 0; i < shape.total(); i++)
			{
				top_blob[i] = bottom_blob[valid_index(i)] / norm;
			}
		}
		if (norm_type == MINMAX)
		{
			float min = FLT_MAX;
			float max = -FLT_MAX;
			for (size_t i = 0; i < shape.total(); i++)
			{
				size_t idx = valid_index(i);
				if (min > bottom_blob[idx]) min = bottom_blob[idx];
				if (max < bottom_blob[idx]) max = bottom_blob[idx];
			}

			float range = (max - min);

			for (size_t i = 0; i < shape.total(); i++)
			{
				top_blob[i] = (bottom_blob[valid_index(i)] - min) / range;
			}
		}
	}

	void Normalize::Set(const std::string& pname, const std::any& val)
	{
		if ("norm_type" == pname && val.type() == typeid(NormType))
		{
			norm_type = std::any_cast<NormType>(val);
		}
	}
}