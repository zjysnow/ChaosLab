#pragma once

#include "core/types.hpp"
#include "core/array.hpp"
#include "core/tensor.hpp"

#ifdef _WIN32
#include <iostream>
#include <format>

namespace chaos
{
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
			for (int i = 0; i < steps[0] * shape[0] * packing; i += static_cast<int>(steps[0] * packing))
			{
				switch (packing)
				{
				case Packing::CHW:
					stream << std::format(", {0}" + 2LL * !i, data[i]);
					break;
				case Packing::C2HW2:
					stream << std::format(", {0} {1}" + 2LL * !i, data[i], data[i + 1]);
					break;
				case Packing::C3HW3:
					stream << std::format(", {0} {1} {2}" + 2LL * !i,
						data[i], data[i + 1], data[i + 2]);
					break;
				case Packing::C4HW4:
					stream << std::format(", {0} {1} {2} {3}" + 2LL * !i,
						data[i], data[i + 1], data[i + 2], data[i + 3]);
					break;
				case Packing::C8HW8:
					stream << std::format(", {0} {1} {2} {3} {4} {5} {6} {7}" + 2LL * !i,
						data[i], data[i + 1], data[i + 2], data[i + 3], data[i + 4], data[i + 5], data[i + 6], data[i + 7]);
					break;
				default:
					LOG(FATAL) << "invalid packing data";
				}
			}
			break;
		case 2:
			for (int i = 0; i < shape[0]; i++)
			{
				PrintTensor<Type>(stream, Tensor(shape[-1], depth, packing, data + i * steps[0] * packing, steps[-1]));
				if (i < shape[0] - 1) stream << std::endl;
			}
			break;
		case 0:
			break;
		default:
			size_t dims = shape.size();
			int h = shape[-2];
			int w = shape[-1];
			size_t csize = static_cast<size_t>(h) * w;
			int estep = steps[-1];
			int rstep = steps[-2];
			int total = shape.total();
			for (size_t i = 0; i < total; i += csize)
			{
				size_t offset = 0;
				size_t idx = i;
				for (int d = 1; d <= dims; d++)
				{
					size_t k = idx % shape[-d];
					offset += k * steps[-d];
					idx /= shape[-d];
				}
				PrintTensor<Type>(stream, Tensor(shape.ranges(-2,-1), depth, packing, data + offset * packing, steps.ranges(-2,-1)));
				if (i < total - csize) stream << ";" << std::endl;
			}
			break;
		}
		stream << "]";
	}

	//CHAOS_API std::ostream& operator<<(std::ostream& stream, const GPUInfo& info);
	//CHAOS_API std::ostream& operator<<(std::ostream& stream, const Depth& depth);
	//CHAOS_API std::ostream& operator<<(std::ostream& stream, const Packing& packing);
	// just for float or uchar
	static std::ostream& operator<<(std::ostream& stream, const Tensor& tensor)
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
#else
namespace chaos
{

}
#endif