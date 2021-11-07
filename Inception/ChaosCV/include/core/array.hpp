#pragma once

#include "core/def.hpp"
#include "core/log.hpp"
#include "core/types.hpp"

#include <vector>
#include <iostream>
#include <memory>
#include <numeric>
#include <type_traits>

namespace chaos
{
	template<class Type>
	using Arithmetic = std::enable_if_t<std::is_integral_v<Type> or std::is_floating_point_v<Type> or std::is_same_v<Complex, Type>, bool>;

	template<class Type>
	using Integral = std::enable_if_t<std::is_integral_v<Type>, bool>;

	template<class Type, Arithmetic<Type> = true>
	class Array
	{
	public:
		Array() = default;
		explicit Array(size_t new_size) { Create(new_size); }
		Array(size_t new_size, Type val) { Create(new_size, &val); }

		Array(size_t new_size, Type* data, size_t inc = 0) { Create(new_size, data, inc); }

		Array(const std::initializer_list<Type>& list)
		{
			Create(list.size(), list.begin(), 1);
		}

		virtual ~Array() { Release(); }

		// copy constructor
		Array(const Array<Type>& arr)
		{
			Create(arr.size_, arr.data_, 1);
		}
		// copy assignment
		Array<Type>& operator=(const Array<Type>& arr)
		{
			// see rule of five https://en.cppreference.com/w/cpp/language/rule_of_three
			return *this = Array<Type>(arr);
		}

		// move constructor
		Array(Array<Type>&& arr) noexcept : size_(std::exchange(arr.size_, 0)), data_(std::exchange(arr.data_, nullptr)) {}
		// move assignment
		Array<Type>& operator=(Array<Type>&& arr) noexcept
		{
			std::swap(size_, arr.size_);
			std::swap(data_, arr.data_);
			return *this;
		}

		void Resize(size_t new_size)
		{
			Array<Type> ori = std::move(*this);

			size_ = new_size;
			data_ = static_cast<Type*>(::operator new(size_ * sizeof(Type), std::align_val_t{ alignof(Type) }));
			for (int i = 0; i < size_; i++)
			{
				std::construct_at(std::addressof(data_[i]), i < ori.size_ ? ori[i] : 0);
			}
		}

		size_t size() const noexcept { return size_; }
		Type* data() const noexcept { return data_; }

		// return arr[a:b]
		// for example, Array<float> arr = {1,2,3,4,5,6,7};
		// then arr.ranges(-5, -1) means [3,4,5,6,7];
		// if out of range, will return random value
		Array<Type> ranges(int a, int b, size_t inc = 1) const
		{
			DCHECK_NE(inc, 0) << "requires delta != 0";
			DCHECK_LE(a, b) << "requires a < b";

			size_t new_size = (1LL + b - a) / inc;
			return Array<Type>(new_size, data_ + (a + size_) % size_, inc);
		}

		Type& operator[](int idx) const noexcept
		{ 
			return data_[(idx + size_) % size_];
		}

		operator std::vector<Type>() const noexcept
		{
			std::vector<Type> vec(size_);
			memcpy(vec.data(), data_, size_ * sizeof(Type));
			return vec;
		}

		//friend class Shape;
	protected:
		void Create(size_t new_size)
		{
			size_ = new_size;
			if (size_ > 0)
			{
				data_ = static_cast<Type*>(::operator new(size_ * sizeof(Type), std::align_val_t{ alignof(Type) }));
				for (size_t i = 0; i < size_; i++)
				{
					std::construct_at(std::addressof(data_[i]), 0);
				}
			}
		}
		void Create(size_t new_size, const Type* data, size_t inc = 0)
		{
			size_ = new_size;
			if (size_ > 0)
			{
				data_ = static_cast<Type*>(::operator new(size_ * sizeof(Type), std::align_val_t{ alignof(Type) }));
				for (size_t i = 0; i < size_; i++, data += inc)
				{
					std::construct_at(std::addressof(data_[i]), *data);
				}
			}
		}

		void Release()
		{
			if (data_)
			{
				for (size_t i = 0; i < size_; i++)
				{
					data_[i].~Type();
				}
				::operator delete (static_cast<void*>(data_), std::align_val_t{ alignof(Type) });
			}
			data_ = nullptr;
			size_ = 0;
		}

		Type* data_ = nullptr;
		size_t size_ = 0;
	};

	// same with tensorflow ranges
	template<class Type, std::enable_if_t<not std::is_same_v<Complex, Type>, bool> = true>
	Array<Type> Range(const Type& start, const Type& limit, const Type& delta = static_cast<Type>(1))
	{
		DCHECK_NE(delta, 0) << "requires delta != 0";
		if (delta > 0) DCHECK_LT(start, limit) << "requires start < limit when delta > 0";
		if (delta < 0) DCHECK_GT(start, limit) << "requires start > limit when delta < 0";

		//[start, limit)
		long double n = static_cast<long double>(limit - start) / delta;
		size_t size = (n == static_cast<size_t>(n) ? n : n + 1);
		Array<Type> ranges(size);
		for (int i = 0; i < size; i++)
		{
			ranges[i] = start + i * delta;
		}
		return ranges;
	}

	template<class Type>
	const Type* begin(const Array<Type>& arr) noexcept
	{
		return arr.data();
	}

	template<class Type>
	const Type* end(const Array<Type>& arr) noexcept
	{
		return arr.data() + arr.size();
	}

	template<class Type>
	static inline bool operator==(const Array<Type>& lhs, const Array<Type>& rhs)
	{
		if (lhs.size() != rhs.size()) return false;
		for (int i = 0; i < lhs.size(); i++)
		{
			if (lhs[i] != rhs[i]) return false;
		}
		return true;
	}

	template<class Type>
	static inline std::ostream& operator<<(std::ostream& stream, const Array<Type>& arr)
	{
		stream << "[" << arr[0];
		for (int i = 1; i < arr.size(); i++)
		{
			stream << ", " << arr[i];
		}
		return stream << "]";
	}


	class CHAOS_API Steps : public Array<int>
	{
	public:
		Steps();
		Steps(int s0);
		Steps(int s0, int s1);
		Steps(int s0, int s1, int s2);

		Steps(Array<int>&& arr) noexcept;

		template<class Type, Integral<Type> = true>
		Steps(const std::initializer_list<Type>& list)
		{
			size_ = list.size();
			data_ = static_cast<int*>(::operator new(size_ * sizeof(int), std::align_val_t{ alignof(int) }));
			for (size_t i = 0; const auto & data : list)
			{
				std::construct_at(std::addressof(data_[i++]), static_cast<int>(data));
			}
		}


	};

	class CHAOS_API Shape : public Array<int>
	{
	public:
		

		Shape();
		Shape(int d0);
		Shape(int d0, int d1);
		Shape(int d0, int d1, int d2);

		Shape(Array<int>&& arr) noexcept;

		template<class Type, Integral<Type> = true>
		Shape(const std::initializer_list<Type>& list)
		{
			size_ = list.size();
			data_ = static_cast<int*>(::operator new(size_ * sizeof(int), std::align_val_t{ alignof(int) }));
			for (size_t i = 0; const auto & data : list)
			{
				std::construct_at(std::addressof(data_[i++]), static_cast<int>(data));
			}
		}

		int total() const noexcept;
		Steps steps() const noexcept;
	};

	CHAOS_API Shape Squeeze(const Shape& shape, const Array<int>& axis = Array<int>());
	CHAOS_API Shape ExpandDims(const Shape& shape, const Array<int>& axis);
}