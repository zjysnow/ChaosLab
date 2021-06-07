#include "core/io.hpp"

namespace chaos
{
	std::ostream& operator<<(std::ostream& stream, const Tensor& tensor)
	{
		switch (tensor.depth)
		{
		case Depth::D1: // as uchar
			PrintTensor<uchar>(stream, tensor);
			break;
		case Depth::D4: 
			PrintTensor<float>(stream, tensor);
			break;
		default:
			LOG(FATAL) << "just for float or uchar type";
			break;
		}
		return stream << std::endl << "<Tensor " << tensor.shape << ">";
	}
}