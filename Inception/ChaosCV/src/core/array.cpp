#include "core/array.hpp"
#include "core/log.hpp"

namespace chaos
{
	Steps::Steps() : Array<int>(0) {}
	Steps::Steps(int s0) : Array<int>(1) { data_[0] = s0; }
	Steps::Steps(int s0, int s1) : Array<int>(2) { data_[0] = s0; data_[1] = s1; }
	Steps::Steps(int s0, int s1, int s2) : Array<int>(3) { data_[0] = s0; data_[1] = s1; data_[2] = s2; }
	Steps::Steps(const Array<int>& arr)
	{
		Create(arr.size(), arr.data(), 1);
	}
	

	Shape::Shape() : Array<int>(0) {}
	Shape::Shape(int d0) : Array<int>(1) { data_[0] = d0; }
	Shape::Shape(int d0, int d1) : Array<int>(2) { data_[0] = d0; data_[1] = d1; }
	Shape::Shape(int d0, int d1, int d2) : Array<int>(3) { data_[0] = d0; data_[1] = d1; data_[2] = d2; }
	Shape::Shape(const Array<int>& arr)
	{
		Create(arr.size(), arr.data(), 1);
	}
	//Shape::Shape(const std::vector<int>& vec)
	//{
	//	Create(vec.size(), vec.data(), 1);
	//}
	int Shape::total() const noexcept { return std::accumulate(data_, data_ + size_, 1, std::multiplies<int>()); }
	Steps Shape::steps() const noexcept
	{
		Steps steps = Array<int>(size_, 1);
		for (size_t i = size_ - 1, j = size_ - 2; i > 0; i--, j--)
		{
			steps[j] = steps[i] * data_[i];
		}
		return steps;
	}

	Shape Squeeze(const Shape& shape, const Array<int>& axis)
	{
		size_t dims = shape.size();
		Array<bool> retain(dims, true);
		if (axis.size() == 0)
		{
			for (size_t i = 0; i < shape.size(); i++)
			{
				if (shape[i] == 1)
				{
					retain[i] = false;
					dims--;
				}
			}
		}
		else
		{
			for (const auto& i : axis)
			{
				if (shape[i] == 1)
				{
					retain[i] = false;
					dims--;
				}
				else
				{
					DLOG(FATAL) << "can not squeeze dim[" << i << "], expcet a dimension of 1, got " << shape[i];
				}
			}
		}

		Shape squeezed = Array<int>(dims);
		for (size_t i = 0, j = 0; i < shape.size(); i++)
		{
			if (retain[i]) squeezed[j++] = shape[i];
		}
		return squeezed;
	}

	Shape ExpandDims(const Shape& shape, const Array<int>& axis)
	{
		size_t dims = shape.size() + axis.size();

		Shape expanded = Array<int>(dims);
		for (const auto& i : axis)
		{
			expanded[i] = 1;
		}

		for (size_t i = 0, j = 0; i < dims; i++)
		{
			if (expanded[i] == 0) expanded[i] = shape[j++];
		}

		return expanded;
	}
}