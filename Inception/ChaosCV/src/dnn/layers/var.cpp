#include "dnn/layers/var.hpp"
#include "utils/op.hpp"

namespace chaos
{
	Var::Var() : Layer("Var") {}

	void Var::Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const
	{
		size_t dims = bottom_blob.shape.dims;
		CHECK_LE(vecdim.size(), dims);
		size_t n = 1; // bottom_blob.shape.total();
		Shape shape = bottom_blob.shape;
		if (all)
		{
			for (size_t i = 0; i < dims; i++)
			{
				n *= shape[i];
				shape[i] = 1;
			}
		}
		else
		{
			for (const auto& i : vecdim)
			{
				CHECK_LE(i, dims) << "out of range";
				n *= shape[i];
				shape[i] = 1;
			}
		}
		Steps steps = shape.steps();

		size_t top_idx = 0;
		size_t bottom_idx = 0;
		auto ReIndex = [&](size_t idx) {
			top_idx = 0;
			bottom_idx = 0;
			for (size_t d = 0; d < dims; d++)
			{
				size_t k = idx % bottom_blob.shape[dims - d - 1];
				top_idx += (k >= shape[dims - d - 1] ? 0 : k) * steps[dims - d - 1];
				bottom_idx += k * bottom_blob.steps[dims - d - 1];
				idx /= bottom_blob.shape[dims - d - 1];
			}
		};

		Tensor mean = Tensor::zeros(shape, opt.workspace_allocator);
		for (size_t i = 0; i < bottom_blob.shape.total(); i++)
		{
			ReIndex(i);
			mean[top_idx] += (bottom_blob[bottom_idx] / n);
		}

		if (top_blob.empty()) top_blob.Create(shape, steps, DataType::D4, Packing::CHW, opt.blob_allocator);
		CHECK_EQ(shape, top_blob.shape);
		memset(top_blob.data, 0, shape.total() * sizeof(float));

		float base = unbiased ? n - 1 : n;
		for (size_t i = 0; i < bottom_blob.shape.total(); i++)
		{
			ReIndex(i);
			top_blob[top_idx] += (bottom_blob[bottom_idx] - mean[top_idx]) * (bottom_blob[bottom_idx] - mean[top_idx]) / base;
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