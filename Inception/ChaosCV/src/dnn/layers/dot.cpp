#include "dnn/layers/dot.hpp"

namespace chaos
{
	Dot::Dot() : Layer("Dot") {}

	void Dot::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const
	{
		CHECK_EQ(2, bottom_blobs.size()) << "layer '" << type << "' expect 2 inputs but got " << bottom_blobs.size();

		const Tensor& a = bottom_blobs[0];
		const Tensor& b = bottom_blobs[1];

		CHECK_EQ(1, a.shape.dims) << "input a must be a vector";
		CHECK_EQ(1, b.shape.dims) << "input b must be a vector";
		CHECK_EQ(a.shape[0], b.shape[0]) << "inputs must have same shape";

		CHECK_EQ(1, top_blobs.size()) << "layer '" << type << "' expect 1 output but got " << top_blobs.size();
		Tensor& c = top_blobs[0];
		if (c.empty()) c.Create(Shape(1u), { 1 }, DataType::D4, Packing::CHW, opt.blob_allocator);
		CHECK_EQ(Shape(1u), c.shape);
		c[0] = 0;

		uint32 n = a.shape[0];
		for (uint32 i = 0; i < n; i++)
		{
			c[0] += (a[i] * b[i]);
		}
	}
}