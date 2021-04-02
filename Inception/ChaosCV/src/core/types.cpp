#include "core/core.hpp"

namespace chaos
{
	Steps::Steps(size_t size) : size(size)
	{
		size_t capacity = AlignSize(size * sizeof(uint32), 4);
		data = (uint32*)FastMalloc(capacity);
	}

	Steps::~Steps()
	{
		if (data)
		{
			FastFree(data);
			data = nullptr;
		}
	}

	Steps::Steps(const Steps& steps) : Steps(steps.size)
	{
		for (size_t i = 0; i < size; i++)
		{
			data[i] = steps[i];
		}
	}
	Steps& Steps::operator=(const Steps& steps)
	{
		if (data) FastFree(data);

		size = steps.size;
		size_t capacity = AlignSize(size * sizeof(uint32), 4);
		data = (uint32*)FastMalloc(capacity);

		for (size_t i = 0; i < size; i++)
		{
			data[i] = steps[i];
		}

		return *this;
	}

	void Steps::Insert(size_t pos, size_t cnt, uint32 val)
	{
		CHECK_LT(pos, size) << "out of range.";

		size += cnt;
		size_t capacity = AlignSize(size * sizeof(uint32), 4);
		uint32* ndata = (uint32*)FastMalloc(capacity);
		// copy data
		for (size_t i = 0; i < (size - cnt); i++)
		{
			if (i < pos) ndata[i] = data[i];
			if (i >= pos) ndata[i + cnt] = data[i];
		}
		// insert new data
		for (size_t i = 0; i < cnt; i++)
		{
			ndata[pos + i] = val;
		}
		if (data) FastFree(data);
		data = ndata;
	}

	bool operator==(const Steps& lhs, const Steps& rhs)
	{
		if (lhs.size != rhs.size) return false;
		for (size_t i = 0; i < lhs.size; i++)
		{
			if (lhs[i] != rhs[i]) return false;
		}
		return true;
	}


	Shape::Shape(uint32 d0) : Shape(1ULL)
	{
		data[0] = d0;
	}
	Shape::Shape(uint32 d0, uint32 d1) : Shape(2ULL)
	{
		data[0] = d0;
		data[1] = d1;
	}
	Shape::Shape(uint32 d0, uint32 d1, uint32 d2) : Shape(3ULL)
	{
		data[0] = d0;
		data[1] = d1;
		data[2] = d2;
	}

	Shape::Shape(size_t dims) : dims(dims)
	{
		size_t capacity = AlignSize(dims * sizeof(uint32), 4);
		data = (uint32*)FastMalloc(capacity);
	}

	Shape::~Shape()
	{
		if (data)
		{
			FastFree(data);
			data = nullptr;
		}
	}

	Shape::Shape(const Shape& shape) : Shape(shape.dims)
	{
		for (size_t i = 0; i < dims; i++)
		{
			data[i] = shape[i];
		}
	}
	Shape& Shape::operator=(const Shape& shape)
	{
		if (data) FastFree(data);

		dims = shape.dims;
		size_t capacity = AlignSize(dims * sizeof(uint32), 4);
		data = (uint32*)FastMalloc(capacity);

		for (size_t i = 0; i < dims; i++)
		{
			data[i] = shape[i];
		}

		return *this;
	}

	void Shape::Insert(size_t pos, size_t cnt, uint32 val)
	{
		CHECK_LT(pos, dims) << "out of range.";

		dims += cnt;
		size_t capacity = AlignSize(dims * sizeof(uint32), 4);
		uint32* ndata = (uint32*)FastMalloc(capacity);
		// copy data
		for (size_t i = 0; i < (dims - cnt); i++)
		{
			if (i < pos) ndata[i] = data[i];
			if (i >= pos) ndata[i + cnt] = data[i];
		}
		// insert new data
		for (size_t i = 0; i < cnt; i++)
		{
			ndata[pos + i] = val;
		}
		if (data) FastFree(data);
		data = ndata;
	}

	Steps Shape::steps() const
	{
		Steps steps_(dims);
		steps_[dims - 1] = 1;
		for (size_t i = 1; i < dims; i++)
		{
			steps_[dims - i - 1] = steps_[dims - i] * data[dims - i];
		}
		return steps_;
	}

	size_t Shape::total() const
	{
		return std::accumulate(data, data + dims, 1, std::multiplies<uint32>());
	}

	bool operator==(const Shape& lhs, const Shape& rhs)
	{
		if (lhs.dims != rhs.dims) return false;
		for (size_t i = 0; i < lhs.dims; i++)
		{
			if (lhs[i] != rhs[i]) return false;
		}
		return true;
	}

	std::wostream& operator<<(std::wostream& stream, const Shape& shape)
	{
		stream << "[";
		for (size_t i = 0; i < shape.dims; i++)
		{
			stream << Format(L",%d" + !i, shape[i]);
		}
		return stream << "]";
	}

}