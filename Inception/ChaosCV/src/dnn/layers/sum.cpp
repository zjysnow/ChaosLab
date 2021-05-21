#include "dnn/layers/sum.hpp"

namespace chaos
{
	namespace dnn
	{
		Sum::Sum() : Layer("Sum") {}

		void Sum::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const
		{
			CHECK_EQ(1, bottom_blobs.size()) << "layer Sum expect 1 input but got " << bottom_blobs.size();
			CHECK_EQ(1, top_blobs.size()) << "layer Sum expect 1 output but got " << top_blobs.size();

			const Tensor& A = bottom_blobs[0];
			size_t dims = A.shape.dims;
			CHECK_LE(vecdim.size(), dims) << "dims of A should greater-equal than size of vecdims";

			Shape shape = A.shape;
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

			Tensor& S = top_blobs[0];
			if (S.empty()) S.Create(shape, steps, DataType::D4, Packing::CHW, opt.blob_allocator);
			CHECK_EQ(shape, S.shape);
			memset(S.data, 0, shape.total() * sizeof(float));

			for (size_t i = 0; i < A.shape.total(); i++)
			{
				size_t s_idx = 0;
				size_t a_idx = 0;
				size_t idx = i;
				for (size_t d = 0; d < dims; d++)
				{
					size_t k = idx % A.shape[dims - d - 1];
					s_idx += (k >= shape[dims - d - 1] ? 0 : k) * steps[dims - d - 1];
					a_idx += k * A.steps[dims - d - 1];
					idx /= A.shape[dims - d - 1];
				}
				S[s_idx] += A[a_idx];
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
}