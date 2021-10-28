#include "core/array.hpp"
#include "..\..\include\core\array.hpp"

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

	
}