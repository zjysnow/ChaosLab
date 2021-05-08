#include "dnn/layers/diag.hpp"

namespace chaos
{
	Diag::Diag() : Layer("Diag") {}

	void Diag::Forward(const Tensor& bottom_blob, Tensor& top_blob, const Option& opt) const
	{
		CHECK_EQ(DataType::D4, bottom_blob.dtype);
		CHECK_LE(1, bottom_blob.shape.dims);
		CHECK_GE(2, bottom_blob.shape.dims) << "input should be a matrix or a vector";

		if (1 == bottom_blob.shape.dims) // vector 2 matrix 
		{
			uint32 n = bottom_blob.shape[0];
			Shape shape = { n,n };
			
			if (top_blob.empty()) top_blob.Create(shape, shape.steps(), DataType::D4, Packing::CHW, opt.blob_allocator);
			CHECK_EQ(shape, top_blob.shape);
			memset(top_blob.data, 0, top_blob.total() * sizeof(float));

			uint32 rstep = top_blob.steps[0];
			for (uint32 i = 0; i < n; i++)
			{
				top_blob[i * rstep + i] = bottom_blob[i];
			}
		}
		else // 2 == bottom_blob.shape.dims; // matrix 2 vector
		{
			CHECK_EQ(bottom_blob.shape[0], bottom_blob.shape[1]);

			uint32 n = bottom_blob.shape[0];
			Shape shape = { n };

			if (top_blob.empty()) top_blob.Create(shape, { 1 }, DataType::D4, Packing::CHW, opt.blob_allocator);
			CHECK_EQ(shape, top_blob.shape);

			uint32 rstep = bottom_blob.steps[0];
			for (uint32 i = 0; i < n; i++)
			{
				top_blob[i] = bottom_blob[i + rstep + i];
			}
		}
	}
}