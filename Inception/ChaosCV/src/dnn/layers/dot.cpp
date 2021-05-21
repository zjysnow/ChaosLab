#include "dnn/layers/dot.hpp"

namespace chaos
{
	namespace dnn
	{
		Dot::Dot() : Layer("Dot") {}

		void Dot::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const
		{
			CHECK_EQ(2, bottom_blobs.size()) << "layer '" << type << "' expect 2 inputs but got " << bottom_blobs.size();

			const Tensor& a = bottom_blobs[0];
			const Tensor& b = bottom_blobs[1];
			CHECK_EQ(a.shape, b.shape) << "input a and b must have same shape";

			size_t dims = a.shape.dims;

			Shape c_shape = a.shape;
			if (all)
			{
				for (size_t i = 0; i < dims; i++) c_shape[i] = 1;
			}
			else
			{
				for (const auto& i : vecdim)
				{
					CHECK_LE(i, dims) << "out of range";
					c_shape[i] = 1;
				}
			}
			Steps c_steps = c_shape.steps();

			CHECK_EQ(1, top_blobs.size()) << "layer '" << type << "' expect 1 output but got " << top_blobs.size();
			Tensor& c = top_blobs[0];
			if (c.empty()) c.Create(c_shape, c_steps, DataType::D4, Packing::CHW, opt.blob_allocator);
			CHECK_EQ(c_shape, c.shape);
			memset(c.data, 0, c.total() * c.dtype);

			for (size_t i = 0; i < a.shape.total(); i++)
			{
				size_t a_idx = 0;
				size_t c_idx = 0;
				size_t idx = i;
				for (size_t d = 0; d < dims; d++)
				{
					size_t k = idx % a.shape[dims - d - 1];
					c_idx += (k >= c_shape[dims - d - 1] ? 0 : k) * c_steps[dims - d - 1];
					a_idx += k * a.steps[dims - d - 1];
					idx /= a.shape[dims - d - 1];
				}
				c[c_idx] += (a[a_idx] * b[a_idx]);
			}
		}

		void Dot::Set(const std::string& pname, const std::any& val)
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
}