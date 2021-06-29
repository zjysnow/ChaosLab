#include "dnn/layers/diag.hpp"

namespace chaos
{
	inline namespace dnn
	{
		Diag::Diag() : Layer("Diag") {}

		void Diag::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const
		{
			CHECK_EQ(1, bottom_blobs.size());
			CHECK_EQ(1, top_blobs.size());

			const Tensor& a = bottom_blobs[0];
			CHECK(1 == a.shape.size() || 2 == a.shape.size());

			Tensor& d = top_blobs[0];

			if (1 == a.shape.size())
			{
				int n = a.shape[0];
				int m = n + std::abs(diagonal);

				if (d.empty()) d.Create(Shape(m, m), Steps(m, 1), Depth::D4, Packing::CHW, opt.blob_allocator);
				CHECK_EQ(Shape(m,m), d.shape);

				memset(d.data, 0, m * m * sizeof(float));
				for (int i = 0; i < n; i++)
				{
					int idx = diagonal > 0 ? i + i * m + diagonal : i + (i - diagonal) * m;
					d[idx] = a[i];
				}
			}
			if (2 == a.shape.size())
			{
				CHECK_EQ(a.shape[0], a.shape[1]);
				int m = a.shape[0];
				int n = m - std::abs(diagonal);

				if (d.empty()) d.Create(Shape(n), Steps(1), Depth::D4, Packing::CHW, opt.blob_allocator);
				CHECK_EQ(Shape(n), d.shape);

				for (int i = 0; i < n; i++)
				{
					int idx = diagonal > 0 ? i + i * m + diagonal : i + (i - diagonal) * m;
					d[i] = a[idx];
				}
			}
		}

		void Diag::Set(const std::string& pname, const std::any& param)
		{
			if ("diagonal" == pname) diagonal = std::any_cast<int>(param);
		}
	}
}