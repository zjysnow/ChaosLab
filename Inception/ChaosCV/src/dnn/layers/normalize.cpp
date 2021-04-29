#include "dnn/layers/normalize.hpp"

namespace chaos
{
	Normalize::Normalize() : Layer("Normalize")
	{

	}

	void Normalize::Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const
	{
		size_t total = bottom_blob.total();
		CHECK_EQ(total, bottom_blob.shape.total());

		top_blob.CreateLike(bottom_blob, opt.blob_allocator);

		if (norm_type == L1)
		{
			float norm = 0.f;
			for (size_t i = 0; i < total; i++)
			{
				norm += bottom_blob[i];
			}
			for (size_t i = 0; i < total; i++)
			{
				top_blob[i] = bottom_blob[i] / norm;
			}
		}
		if (norm_type == L2)
		{
			float norm = 0.f;
			for (size_t i = 0; i < total; i++)
			{
				norm += bottom_blob[i] * bottom_blob[i];
			}
			norm = std::sqrt(norm);

			for (size_t i = 0; i < total; i++)
			{
				top_blob[i] = bottom_blob[i] / norm;
			}
		}
		if (norm_type == MINMAX)
		{
			float min = FLT_MAX;
			float max = -FLT_MAX;
			for (size_t i = 0; i < total; i++)
			{
				if (min > bottom_blob[i]) min = bottom_blob[i];
				if (max < bottom_blob[i]) max = bottom_blob[i];
			}

			float range = (max - min);

			for (size_t i = 0; i < total; i++)
			{
				top_blob[i] = (bottom_blob[i] - min) / range;
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