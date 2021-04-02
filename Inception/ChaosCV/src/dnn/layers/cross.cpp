#include "dnn/layers/cross.hpp"

namespace chaos
{
	Cross::Cross() : Layer(L"Cross") {}

	void Cross::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const
	{
		CHECK_EQ(2, bottom_blobs.size()) << "layer '" << type << "' expect 2 inputs but got " << bottom_blobs.size();

		const Tensor& a = bottom_blobs[0];
		const Tensor& b = bottom_blobs[1];

		CHECK_EQ(1, a.shape.dims) << "input a must be a vector";
		CHECK_EQ(1, b.shape.dims) << "input b must be a vector";
		CHECK_EQ(3, a.shape[0]) << "input a must be a 3d vector";
		CHECK_EQ(3, b.shape[0]) << "input b must be a 3d vector";

		CHECK_EQ(1, top_blobs.size()) << "layer '" << type << "' expect 1 output but got " << top_blobs.size();
		Tensor& cross = top_blobs[0];
		cross.Create(Shape(3u), Steps{ 1u }, DataType::D4, Packing::CHW, opt.blob_allocator);

		cross[0] = a[1] * b[2] - a[2] * b[1];
		cross[1] = a[2] * b[0] - a[0] * b[2];
		cross[2] = a[0] * b[1] - a[1] * b[0];
	}
}