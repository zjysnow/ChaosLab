#include "dnn/layers/abs.hpp"

namespace chaos
{
	Abs::Abs() : Layer("Abs") 
	{ 
		support_inplace = true;
	}

	void Abs::Forward(std::vector<Tensor>& bottom_top_blobs) const
	{
		CHECK_EQ(1, bottom_top_blobs.size()) << "layer Abs expect 1 input/output but got " << bottom_top_blobs.size();
		Tensor& val = bottom_top_blobs[0];

		for (size_t i = 0; i < val.total(); i++)
		{
			val[i] = std::abs(val[i]);
		}
	}
}