#include "dnn/layers/sum.hpp"

namespace chaos
{
	Sum::Sum() : Layer("Sum") {}

	void Sum::Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const
	{
		size_t dims = bottom_blob.shape.dims;
		CHECK_LE(vecdim.size(), dims);

		Shape shape = bottom_blob.shape;
		if (all)
		{
			for (size_t i = 0; i < dims; i++) shape[i] = 1;
		}
		else
		{
			for (const auto& i : vecdim)
			{
				CHECK_LE(i, dims) << "out of range";
				shape[i] = 1;
			}
		}
		Steps steps = shape.steps();

		if (top_blob.empty()) top_blob.Create(shape, steps, DataType::D4, Packing::CHW, opt.blob_allocator);
		CHECK_EQ(shape, top_blob.shape);
		memset(top_blob.data, 0, shape.total() * sizeof(float));

		for (size_t i = 0; i < bottom_blob.shape.total(); i++)
		{
			size_t top_idx = 0;
			size_t bottom_idx = 0;
			size_t idx = i;
			for (size_t d = 0; d < dims; d++)
			{
				size_t k = idx % bottom_blob.shape[dims - d - 1];
				top_idx += (k >= shape[dims - d - 1] ? 0 : k) * steps[dims - d - 1];
				bottom_idx += k * bottom_blob.steps[dims - d - 1];
				idx /= bottom_blob.shape[dims - d - 1];
			}
			top_blob[top_idx] += bottom_blob[bottom_idx];
		}
	}

	void Sum::Set(const std::string& pname, const std::any& val)
	{
		if ("vecdim" == pname)
		{
			vecdim = std::any_cast<std::vector<uint32>>(val);
		}
		if ("all" == pname)
		{
			all = std::any_cast<bool>(val);
		}
	}
}