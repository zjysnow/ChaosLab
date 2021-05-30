#pragma once

#include "core/def.hpp"
#include "core/gpu.hpp"
#include "core/tensor.hpp"

#include <format>

namespace chaos
{
	CHAOS_API std::ostream& operator<<(std::ostream& stream, const GPUInfo& info);


	template<class Type>
	void PrintTensor(const Tensor& tensor)
	{
		const Shape& shape = tensor.shape;
		const Steps& steps = tensor.steps;
		int packing = static_cast<int>(tensor.packing);
		Type* data = static_cast<Type*>(tensor.data);
		std::cout << "[";
		switch (shape.size())
		{
		case 1:
			for (uint32 i = 0; i < shape[0] * packing; i+=packing)
			{
				switch (packing)
				{
				case 1:
					std::cout << std::format(", {}" + 2 * !i, data[i]);
					break;
				case 2:
					std::cout << std::format(", {} {}" + 2 * !i, data[i], data[i+1]);
					break;
				case 3:
					std::cout << std::format(", {} {} {}" + 2 * !i, data[i], data[i + 1], data[i + 2]);
					break;
				case 4:
					std::cout << std::format(", {} {}" + 2 * !i, data[i], data[i + 1]);
					break;
				case 8:
					std::cout << std::format(", {} {}" + 2 * !i, data[i], data[i + 1]);
					break;
				default:
					LOG(FATAL) << "invalid packing data";
				}
				 //Print(data, i, i);
			}
			break;
		case 2:
			for (uint32 i = 0; i < shape[0]; i++)
			{
				PrintTensor<Type>(Tensor(Shape(shape[1]), tensor.depth, tensor.packing, data + i * steps[0] * packing));
				if (i < shape[0] - 1) std::cout << std::endl;
			}
			break;
		default:
			size_t dims = shape.size();
			uint32 h = shape[dims - 2];
			uint32 w = shape[dims - 1];
			uint32 rstep = steps[dims - 2];
			for (size_t i = 0; i < shape.total(); i += h * w)
			{
				size_t offset = 0;
				size_t idx = i;
				for (size_t d = 0; d < dims; d++)
				{
					size_t k = idx % shape[dims - d - 1];
					offset += k * steps[dims - d - 1];
					idx /= shape[dims - d - 1];
				}
				PrintTensor<Type>(Tensor(Shape(h, w), tensor.depth, tensor.packing, data + offset * packing, Steps(rstep, 1)));
				if (i < shape.total() - h * w) std::cout << ";" << std::endl;
			}
			break;
		}
		std::cout << "]";
	}
}