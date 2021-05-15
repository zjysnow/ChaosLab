#include "dnn/layers/var.hpp"
#include "utils/op.hpp"

namespace chaos
{
	Var::Var() : Layer("Var") {}

	void Var::Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const
	{
		size_t dims = bottom_blob.shape.dims;
		CHECK_LE(vecdim.size(), dims);
		size_t n = bottom_blob.shape.total();
		if (all)
		{
			if (top_blob.empty()) top_blob.Create(Shape(1u), Steps(1u), DataType::D4, Packing::CHW, opt.blob_allocator);
			CHECK_EQ(Shape(1u), top_blob.shape);
			float mu = 0.f;
			for (size_t i = 0; i < n; i++)
			{
				size_t bottom_idx = 0;
				size_t idx = i;
				for (size_t d = 0; d < dims; d++)
				{
					size_t k = idx % bottom_blob.shape[dims - d - 1];
					bottom_idx += k * bottom_blob.steps[dims - d - 1];
					idx /= bottom_blob.shape[dims - d - 1];
				}
				mu += bottom_blob[bottom_idx];
			}
			mu /= n;

			for (size_t i = 0; i < n; i++)
			{
				size_t bottom_idx = 0;
				size_t idx = i;
				for (size_t d = 0; d < dims; d++)
				{
					size_t k = idx % bottom_blob.shape[dims - d - 1];
					bottom_idx += k * bottom_blob.steps[dims - d - 1];
					idx /= bottom_blob.shape[dims - d - 1];
				}
				top_blob[0] += (bottom_blob[bottom_idx] - mu) * (bottom_blob[bottom_idx] - mu);
			}
			top_blob[0] /= (n - 1);
		}
		else
		{
			Shape shape = bottom_blob.shape;
			size_t np = 1;
			for (const auto& i : vecdim)
			{
				CHECK_LE(i, dims) << "out of range";
				np *= shape[i];
				shape[i] = 1;
			}
			Steps steps = shape.steps();

			if (top_blob.empty()) top_blob.Create(shape, steps, DataType::D4, Packing::CHW, opt.blob_allocator);
			CHECK_EQ(shape, top_blob.shape);
			memset(top_blob.data, 0, shape.total() * sizeof(float));

			Tensor mu = Tensor::zeros(shape, opt.blob_allocator);

			for (size_t i = 0; i < n; i++)
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
				mu[top_idx] += (bottom_blob[bottom_idx] / np);
			}

			for (size_t i = 0; i < n; i++)
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
				top_blob[top_idx] += (bottom_blob[bottom_idx] - mu[top_idx]) * (bottom_blob[bottom_idx] - mu[top_idx]) / (np - 1);
			}
		}
	}

	void Var::Set(const std::string& pname, const std::any& val)
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