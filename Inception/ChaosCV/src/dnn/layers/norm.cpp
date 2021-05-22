#include "dnn/layers/norm.hpp"

namespace chaos
{
	namespace dnn
	{
		Norm::Norm() : Layer("Norm") {}

		void Norm::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const
		{
			CHECK_EQ(1, bottom_blobs.size()) << "layer 'Norm' expect 1 input but got " << bottom_blobs.size();
			CHECK_EQ(1, top_blobs.size()) << "layer 'Norm' expect 1 output but got " << top_blobs.size();

			size_t dims = bottom_blobs[0].shape.dims;
			CHECK(1 == dims || 2 == dims) << "input should be a vector or a matrix";
			
			if (1 == dims)
			{
				const Tensor& v = bottom_blobs[0];
				Tensor& n = top_blobs[0];

				if (n.empty()) n.Create(Shape(1), Steps(1), DataType::D4, Packing::CHW, opt.blob_allocator);
				CHECK_EQ(Shape(1), n.shape);
				
				if (isfinite(p))
				{
					n[0] = 0;
					for (uint32 i = 0; i < v.shape[0]; i++)
					{
						n[0] += std::pow(std::abs(v[i]), p);
					}
					n[0] = std::pow(n[0], 1.f / p);
				}
				else
				{
					if (std::signbit(p)) // p == -INFINITY
					{
						n[0] = FLT_MAX;
						for (uint32 i = 0; i < v.shape[0]; i++)
						{
							n[0] = std::min(n[0], std::abs(v[i]));
						}
					}
					else
					{
						n[0] = -FLT_MAX;
						for (uint32 i = 0; i < v.shape[0]; i++)
						{
							n[0] = std::max(n[0], std::abs(v[i]));
						}
					}
				}
			}
			else // 2 == dims
			{

			}
		}

		void Norm::Set(const std::string& pname, const std::any& val)
		{
			if ("p" == pname)
			{
				p = std::any_cast<float>(val);
			}
		}
	}
}