#include "core/array.hpp"
#include "core/log.hpp"
#include "core/tensor.hpp"
#include "dnn/option.hpp"

#include <format>
#include <numeric>

namespace chaos
{
	template<Arithmetic Type>
	VulkanTensor Array<Type>::Upload(const Option& opt) const
	{
		VulkanTensor up = VulkanTensor((int)size_, Depth::D4, Packing::CHW, opt.staging_vkallocator);
		CHECK(up.allocator->mappable);
		memcpy(up.mapped_data(), data_, sizeof(Type) * size_);
		return up;
	}

	Steps::Steps(int s0) : Array<uint32>(1) { data_[0] = s0; }
	Steps::Steps(int s0, int s1) : Array<uint32>(2) { data_[0] = s0; data_[1] = s1; }
	void Steps::Expand(size_t axis, int dims, uint32 step)
	{
		Steps old_steps = *this; // copy old data
		Resize(size_ + dims, step);
		for (size_t i = 0; i < old_steps.size(); i++)
		{
			if (i < axis) data_[i] = old_steps[i];
			if (i >= axis) data_[i + dims] = old_steps[i];
		}
	}
	bool Steps::operator==(const Steps& rhs)
	{
		if (size_ != rhs.size_) return false;
		for (size_t i = 0; i < size_; i++)
		{
			if (data_[i] != rhs[i]) return false;
		}
		return true;
	}

	Shape::Shape(int d0) : Array<uint32>(1) { data_[0] = d0; }
	Shape::Shape(int d0, int d1) : Array<uint32>(2) 
	{
		data_[0] = d0;
		data_[1] = d1;
	}
	Shape::Shape(int d0, int d1, int d2) : Array<uint32>(3)
	{
		data_[0] = d0;
		data_[1] = d1;
		data_[2] = d2;
	}
	void Shape::Expand(size_t axis, int dims)
	{
		if (dims == 0) return;

		Shape old_shape = *this; // copy old data
		Resize(size_ + dims, 1);
		for (size_t i = 0; i < old_shape.size(); i++)
		{
			if (i < axis) data_[i] = old_shape[i];
			if (i >= axis) data_[i + dims] = old_shape[i];
		}
	}
	Steps Shape::steps() const
	{
		size_t dims = size();
		Steps steps_;
		steps_.Resize(dims, 1);
		for (size_t i = 1; i < dims; i++)
		{
			steps_[dims - i - 1] = steps_[dims - i] * data_[dims - i];
		}
		return steps_;
	}
	uint32 Shape::total() const
	{
		return std::accumulate(data_, data_ + size_, 1, std::multiplies<uint32>());
	}
	bool Shape::operator==(const Shape& rhs)
	{
		if (size_ != rhs.size_) return false;
		for (size_t i = 0; i < size_; i++)
		{
			if (data_[i] != rhs[i]) return false;
		}
		return true;
	}

	Shape operator&(const Shape& lhs, const Shape& rhs)
	{
		Shape a = lhs, b = rhs;
		int a_size = (int)a.size(), b_size = (int)b.size();
		a.Expand(0, std::max(0, b_size - a_size));
		b.Expand(0, std::max(0, a_size - b_size));

		for (size_t i = 0; i < a.size(); i++)
		{
			if (a[i] == 1)
			{
				a[i] = b[i];
			}
			else
			{
				CHECK(a[i] == b[i] || b[i] == 1) << "can not broadcast";
			}
		}
		return a;
	}
	std::ostream& operator<<(std::ostream& stream, const Shape& shape)
	{
		stream << "(";
		for (size_t i = 0; i < shape.size_; i++)
		{
			stream << std::format("x{}"+!i, shape[i]);
		}
		return stream << ")";
	}



}