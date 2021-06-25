#include "dnn/layers/permute.hpp"

namespace chaos
{
	inline namespace dnn
	{
		Permute::Permute() : Layer("Permute") {}

		void Permute::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const
		{
            CHECK_EQ(1, bottom_blobs.size()) << "layer 'Permute' expect 1 input but got " << bottom_blobs.size();
            const Tensor& A = bottom_blobs[0];
            int dims = (int)A.shape.size();
            CHECK_EQ(orders.size(), dims) << "num axes expect " << orders.size() << " but got " << dims;

            bool need_permute = false;
            for (int i = 0; i < dims; i++)
            {
                if (i != orders[i])
                {
                    need_permute = true;
                    break;
                }
            }

            CHECK_EQ(1, top_blobs.size()) << "layer 'Permute' expect 1 output but got " << top_blobs.size();
            Tensor& P = top_blobs[0];
            Shape shape = A.shape;
            if (not need_permute)
            {
                if (P.empty()) P.Create(shape, shape.steps(), Depth::D4, Packing::CHW, opt.blob_allocator);
                CHECK_EQ(shape, P.shape) << "expect " << shape << " but got " << P.shape;
                A.CopyTo(P);
            }
            else
            {
                for (int i = 0; i < dims; i++) shape[i] = A.shape[orders[i]];
                if (P.empty()) P.Create(shape, shape.steps(), Depth::D4, Packing::CHW, opt.blob_allocator);
                CHECK_EQ(shape, P.shape) << "expect " << shape << " but got " << P.shape;

                for (size_t i = 0; i < shape.total(); i++)
                {
                    size_t a_idx = 0;
                    size_t p_idx = 0;
                    size_t idx = i;
                    for (int d = dims - 1; d >= 0; d--)
                    {
                        uint32 order = orders[d];
                        size_t k = idx % shape[d];
                        p_idx += k * top_blobs[0].steps[d];
                        a_idx += k * bottom_blobs[0].steps[order];
                        idx /= shape[d];
                    }
                    P[p_idx] = A[a_idx];
                }
            }
		}

		void Permute::Set(const std::string& pname, const std::any& val)
		{
			if ("orders" == pname) orders = std::any_cast<Array<int>>(val);
		}
	}
}