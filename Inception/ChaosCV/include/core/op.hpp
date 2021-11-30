#pragma once

#include "core/def.hpp"
#include "core/types.hpp"
#include "core/array.hpp"

#include <numeric>
#include <algorithm>

namespace chaos
{
	template<class Type, Arithmetic<Type> = true>
	struct Add { Type operator()(const Type& lhs, const Type& rhs) { return lhs + rhs; } };

	template<class Type, Arithmetic<Type> = true>
	struct Sub { Type operator()(const Type& lhs, const Type& rhs) { return lhs - rhs; } };

	template<class Type, Arithmetic<Type> = true>
	struct Mul { Type operator()(const Type& lhs, const Type& rhs) { return lhs * rhs; } };

	template<class Type, Arithmetic<Type> = true>
	struct Div { Type operator()(const Type& lhs, const Type& rhs) { return lhs / rhs; } };

	// can add other binary operators

	template<class Type, class Op>
	Array<Type> BinaryOp(const Array<Type>& lhs, const Array<Type>& rhs)
	{
		Op op;
		size_t size = lhs.size();
		DCHECK_EQ(size, rhs.size());
		
		Array<Type> arr = Array<Type>(size);
		for (size_t i = 0; i < size; i++)
		{
			arr[i] = op(lhs[i], rhs[i]);
		}
		return arr;
	}
	template<class Type, class Op>
	Array<Type> BinaryOp(const Array<Type>& lhs, const Type& rhs)
	{
		Op op;
		size_t size = lhs.size();
		Array<Type> arr = Array<Type>(size);
		for (size_t i = 0; i < size; i++)
		{
			arr[i] = op(lhs[i], rhs);
		}
		return arr;
	}
	template<class Type, class Op>
	Array<Type> BinaryOp(const Type& lhs, const Array<Type>& rhs)
	{
		Op op;
		size_t size = rhs.size();
		Array<Type> arr = Array<Type>(size);
		for (size_t i = 0; i < size; i++)
		{
			arr[i] = op(lhs, rhs[i]);
		}
		return arr;
	}

	template<class Type>
	Array<Type> operator+(const Array<Type>& lhs, const Array<Type>& rhs)
	{
		return BinaryOp<Type, Add<Type>>(lhs, rhs);
	}
	template<class Type>
	Array<Type> operator+(const Array<Type>& lhs, const Type& rhs)
	{
		return BinaryOp<Type, Add<Type>>(lhs, rhs);
	}
	template<class Type>
	Array<Type> operator+(const Type& lhs, const Array<Type>& rhs)
	{
		return BinaryOp<Type, Add<Type>>(lhs, rhs);
	}

	template<class Type>
	Array<Type> operator-(const Array<Type>& lhs, const Array<Type>& rhs)
	{
		return BinaryOp<Type, Sub<Type>>(lhs, rhs);
	}
	template<class Type>
	Array<Type> operator-(const Array<Type>& lhs, const Type& rhs)
	{
		return BinaryOp<Type, Sub<Type>>(lhs, rhs);
	}
	template<class Type>
	Array<Type> operator-(const Type& lhs, const Array<Type>& rhs)
	{
		return BinaryOp<Type, Sub<Type>>(lhs, rhs);
	}

	template<class Type>
	Array<Type> operator*(const Array<Type>& lhs, const Array<Type>& rhs)
	{
		return BinaryOp<Type, Mul<Type>>(lhs, rhs);
	}
	template<class Type>
	Array<Type> operator*(const Array<Type>& lhs, const Type& rhs)
	{
		return BinaryOp<Type, Mul<Type>>(lhs, rhs);
	}
	template<class Type>
	Array<Type> operator*(const Type& lhs, const Array<Type>& rhs)
	{
		return BinaryOp<Type, Mul<Type>>(lhs, rhs);
	}

	template<class Type>
	Array<Type> operator/(const Array<Type>& lhs, const Array<Type>& rhs)
	{
		return BinaryOp<Type, Div<Type>>(lhs, rhs);
	}
	template<class Type>
	Array<Type> operator/(const Array<Type>& lhs, const Type& rhs)
	{
		return BinaryOp<Type, Div<Type>>(lhs, rhs);
	}
	template<class Type>
	Array<Type> operator/(const Type& lhs, const Array<Type>& rhs)
	{
		return BinaryOp<Type, Div<Type>>(lhs, rhs);
	}

	template<class Type>
	Type sum(const Array<Type>& arr)
	{
		return std::accumulate(begin(arr), end(arr), Type());
	}

	template<class Type>
	Type dot(const Array<Type>& lhs, const Array<Type> rhs)
	{
		return sum(lhs * rhs);
	}

	template<class Type>
	Array<Type> cross(const Array<Type>& lhs, const Array<Type>& rhs)
	{
		DCHECK_EQ(lhs.size(), 3);
		DCHECK_EQ(rhs.size(), 3);

		Array<Type> arr = Array<Type>(3);
		// lhs a = [x1, y1, z2]
		// rhs b = [x2, y2, z2]
		// a x b = [y1*z2-y2*z1, z1*x2-z2*x1, x1*y2-x2*y1]
		arr[0] = lhs[1] * rhs[2] - rhs[1] * lhs[2]; //y1 * z2 - y2 * z1;
		arr[1] = lhs[2] * rhs[0] - rhs[2] * lhs[0]; //z1 * x2 - z2 * x1;
		arr[2] = lhs[0] * rhs[1] - rhs[0] * lhs[1]; //x1 * y2 - x2 * y1;

		return arr;
	}
}