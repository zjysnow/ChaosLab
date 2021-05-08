#include "dnn/layers/normalize.hpp"

#include "core/buffer.hpp"

namespace chaos
{
	Normalize::Normalize() : Layer("Normalize") {}

	void L2Norm(const Tensor& src, size_t* src_index, Tensor& dst,  size_t* dst_index)
	{
		size_t total = src.shape.total();
		float norm = 0.f;
		for (size_t i = 0; i < total; i++)
		{
			norm += src[src_index[i]] * src[src_index[i]];
		}
		norm = std::sqrt(norm);

		for (size_t i = 0; i < total; i++)
		{
			dst[dst_index[i]] = src[src_index[i]] / norm;
		}
	}

	void L1Norm(const Tensor& src, size_t* src_index, Tensor& dst, size_t* dst_index)
	{
		size_t total = src.shape.total();
		float norm = 0.f;
		for (size_t i = 0; i < total; i++)
		{
			norm += src[src_index[i]];
		}

		for (size_t i = 0; i < total; i++)
		{
			dst[dst_index[i]] = src[src_index[i]] / norm;
		}
	}

	void MinMaxNorm()
	{

	}

	void Normalize::Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const
	{
		const Shape shape = bottom_blob.shape;
		size_t dims = shape.dims;
		size_t cnt = shape.total();

		if (top_blob.empty()) top_blob.Create(shape, shape.steps(), DataType::D4, Packing::CHW, opt.blob_allocator);
		CHECK_EQ(shape, top_blob.shape);

		AutoBuffer<size_t> indexes(2 * cnt);
		for (size_t i = 0; i < shape.total(); i++)
		{
			size_t src_idx = 0;
			size_t dst_idx = 0;
			size_t idx = i;
			for (size_t d = 0; d < dims; d++)
			{
				size_t k = idx % shape[dims - d - 1];
				src_idx += k * bottom_blob.steps[dims - d - 1];
				dst_idx += k * top_blob.steps[dims - d - 1];
				idx /= shape[dims - d - 1];
			}
			indexes[i] = src_idx;
			indexes[i + cnt] = dst_idx;
		}

		float norm = 0.f;

		if (norm_type == L2)
		{
			for (size_t i = 0; i < cnt; i++)
			{
				norm += bottom_blob[indexes[i]] * bottom_blob[indexes[i]];
			}
			norm = std::sqrt(norm);
		}

		for (size_t i = 0; i < cnt; i++)
		{
			top_blob[indexes[i + cnt]] = bottom_blob[indexes[i]] / norm;
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