#pragma once

#include "core/def.hpp"
#include "core/gpu.hpp"
#include "core/types.hpp"
#include "core/tensor.hpp"

#include <format>

namespace chaos
{
	CHAOS_API std::ostream& operator<<(std::ostream& stream, const GPUInfo& info);


	template<class Type = float>
	void PrintTensor(std::ostream& stream, const Tensor& tensor)
	{
		const Shape& shape = tensor.shape;
		const Steps& steps = tensor.steps;
		const Depth& depth = tensor.depth;
		const Packing& packing = tensor.packing;
		Type* data = static_cast<Type*>(tensor.data);
		stream << "[";
		switch (shape.size())
		{
		case 1:
			for (uint32 i = 0; i < shape[0] * packing; i += static_cast<int>(packing))
			{
				switch (packing)
				{
				case Packing::CHW:
					stream << std::format(", {0}" + 2 * !i, data[i]);
					break;
				case Packing::C2HW2:
					stream << std::format(", {0} {1}" + 2 * !i, data[i], data[i+1]);
					break;
				case Packing::C3HW3:
					stream << std::format(", {0} {1} {2}" + 2 * !i,
						data[i], data[i + 1], data[i + 2]);
					break;
				case Packing::C4HW4:
					stream << std::format(", {0} {1} {2} {3}" + 2 * !i,
						data[i], data[i + 1], data[i + 2], data[i + 3]);
					break;
				case Packing::C8HW8:
					stream << std::format(", {0} {1} {2} {3} {4} {5} {6} {7}" + 2 * !i,
						data[i], data[i + 1], data[i + 2], data[i + 3], data[i + 4], data[i + 5], data[i + 6], data[i + 7]);
					break;
				default:
					LOG(FATAL) << "invalid packing data";
				}
			}
			break;
		case 2:
			for (uint32 i = 0; i < shape[0]; i++)
			{
				PrintTensor<Type>(stream, Tensor(Shape(shape[1]), depth, packing, data + i * steps[0] * packing));
				if (i < shape[0] - 1) stream << std::endl;
			}
			break;
		default:
			size_t dims = shape.size();
			uint32 h = shape[dims - 2];
			uint32 w = shape[dims - 1];
			uint32 rstep = steps[dims - 2];
			uint32 total = shape.total();
			for (size_t i = 0; i < total; i += h * w)
			{
				size_t offset = 0;
				size_t idx = i;
				for (size_t d = 0; d < dims; d++)
				{
					size_t k = idx % shape[dims - d - 1];
					offset += k * steps[dims - d - 1];
					idx /= shape[dims - d - 1];
				}
				PrintTensor<Type>(stream, Tensor(Shape(h, w), depth, packing, data + offset * packing, Steps(rstep, 1)));
				if (i < total - h * w) stream << ";" << std::endl;
			}
			break;
		}
		stream << "]";
	}
}