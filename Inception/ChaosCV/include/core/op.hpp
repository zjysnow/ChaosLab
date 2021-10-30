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

	template<class Type, class Op>
	Array<Type> BinaryOp(const Array<Type>& lhs, const Array<Type>& rhs)
	{
		Op op;
		DCHECK_EQ(lhs.size(), rhs.size());
		size_t size = lhs.size();
		Array<Type> result = Array<Type>(size);
		for (size_t i = 0; i < size; i++)
		{
			result[i] = op(lhs[i], rhs[i]);
		}
		return result;
	}
	template<class Type, class Op>
	Array<Type> BinaryOp(const Array<Type>& lhs, const Type& rhs)
	{
		Op op;
		size_t size = lhs.size();
		Array<Type> result = Array<Type>(size);
		for (size_t i = 0; i < size; i++)
		{
			result[i] = op(lhs[i], rhs);
		}
		return result;
	}

	template<class Type>
	Array<Type> operator+(const Array<Type>& lhs, const Array<Type>& rhs)
	{
		return BinaryOp<Type, Add<Type>>(lhs, rhs);
	}

	template<class Type>
	Array<Type> operator-(const Array<Type>& lhs, const Array<Type>& rhs)
	{
		return BinaryOp<Type, Sub<Type>>(lhs, rhs);
	}

	template<class Type>
	Array<Type> operator*(const Array<Type>& lhs, const Array<Type>& rhs)
	{
		return BinaryOp<Type, Mul<Type>>(lhs, rhs);
	}
	template<class Type>
	Array<Type> operator*(const Type& lhs, const Array<Type>& rhs)
	{
		return BinaryOp<Type, Mul<Type>>(rhs, lhs);
	}
	template<class Type>
	Array<Type> operator*(const Array<Type>& lhs, const Type& rhs)
	{
		return BinaryOp<Type, Mul<Type>>(lhs, rhs);
	}

	template<class Type>
	Array<Type> operator/(const Array<Type>& lhs, const Array<Type>& rhs)
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
}