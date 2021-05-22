#include "dnn/layers/normalize.hpp"
#include "utils/op.hpp"

namespace chaos
{
	namespace dnn
	{
		Normalize::Normalize() : Layer("Normalize") {}

		void Normalize::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const
		{
			CHECK_EQ(1, bottom_blobs.size());
			CHECK_EQ(1, top_blobs.size());

			const Tensor& A = bottom_blobs[0];
			Tensor& N = top_blobs[0];
			if (N.empty()) N.Create(A.shape, A.shape.steps(), DataType::D4, Packing::CHW, opt.blob_allocator);
			CHECK_EQ(A.shape, N.shape);
			switch (method)
			{
			case ZSCORE:
				break;
			case NORM:
			{
				div(A, norm(A), N);
				break;
			}
			default:
				LOG(FATAL);
				break;
			}
		}
		
		void Normalize::Set(const std::string& pname, const std::any& val)
		{
			if ("method" == pname)
			{
				method = std::any_cast<Method>(val);
			}
		}

	}
}