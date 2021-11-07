#include "core/array.hpp"
#include "core/log.hpp"

namespace chaos
{
	Steps::Steps() : Array<int>(0) {}
	Steps::Steps(int s0) : Array<int>(1) { data_[0] = s0; }
	Steps::Steps(int s0, int s1) : Array<int>(2) { data_[0] = s0; data_[1] = s1; }
	Steps::Steps(int s0, int s1, int s2) : Array<int>(3) { data_[0] = s0; data_[1] = s1; data_[2] = s2; }
	Steps::Steps(Array<int>&& arr) noexcept : Array<int>(arr) {}

	Shape::Shape() : Array<int>(0) {}
	Shape::Shape(int d0) : Array<int>(1) { data_[0] = d0; }
	Shape::Shape(int d0, int d1) : Array<int>(2) { data_[0] = d0; data_[1] = d1; }
	Shape::Shape(int d0, int d1, int d2) : Array<int>(3) { data_[0] = d0; data_[1] = d1; data_[2] = d2; }
	Shape::Shape(Array<int>&& arr) noexcept : Array<int>(arr) {}

	int Shape::total() const noexcept { return std::accumulate(data_, data_ + size_, 1, std::multiplies<int>()); }
	Steps Shape::steps() const noexcept
	{
		Steps steps = Array<int>(size_, 1);
		for (int i = 1, j = 2; j <= size_; i++, j++)
		{
			// steps[-j] steps[-i] * shape[-i];
			steps[-j] = steps[-i] * operator[](-i);
		}
		return steps;
	}

	Shape Squeeze(const Shape& shape, const Array<int>& axis)
	{
		size_t dims = shape.size();

		Array<bool> retain(dims, true);
		if (axis.size() == 0)
		{
			for (int i = 0; i < shape.size(); i++)
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
			DCHECK_LE(axis.size(), dims);
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

		if (dims == 0)
		{
			return Shape(1);
		}
		else
		{
			Shape squeezed = Array<int>(dims);
			for (int i = 0, j = 0; i < shape.size(); i++)
			{
				if (retain[i]) squeezed[j++] = shape[i];
			}
			return squeezed;
		}
	}

	Shape ExpandDims(const Shape& shape, const Array<int>& axis)
	{
		size_t dims = shape.size() + axis.size();

		Shape expanded = Array<int>(dims);
		for (const auto& i : axis)
		{
			expanded[i] = 1;
		}

		for (int i = 0, j = 0; i < dims; i++)
		{
			if (expanded[i] == 0) expanded[i] = shape[j++];
		}

		return expanded;
	}
}