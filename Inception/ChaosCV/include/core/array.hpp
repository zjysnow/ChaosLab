#pragma once

#include "core/def.hpp"
#include "core/types.hpp"
#include "core/allocator.hpp"

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

		const Type& operator[](size_t idx) const noexcept { return data_[idx]; }
		Type& operator[](size_t idx) noexcept { return data_[idx]; }

		size_t size() const noexcept { return size_; }
		const Type* data() const noexcept { return data_; }

		void Resize(size_t new_size)
		{
			Array<Type> ori = std::move(*this);

			size_ = new_size;
			data_ = static_cast<Type*>(::operator new(size_ * sizeof(Type), std::align_val_t{ alignof(Type) }));
			for (size_t i = 0; i < size_; i++)
			{
				std::construct_at(std::addressof(data_[i]), i < ori.size_ ? ori[i] : 0);
			}
		}
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

	template<class Type>
	const Type* begin(const Array<Type>& arr)
	{
		return arr.data();
	}

	template<class Type>
	const Type* end(const Array<Type>& arr)
	{
		return arr.data() + arr.size();
	}


	class CHAOS_API Steps : public Array<int>
	{
	public:
		Steps();
		Steps(int s0);
		Steps(int s0, int s1);
		Steps(int s0, int s1, int s2);

		Steps(const Array<int>& arr);

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
		Shape(const Array<int>& arr);

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

	static inline bool operator==(const Shape& lhs, const Shape& rhs)
	{
		if (lhs.size() != rhs.size()) return false;
		for (size_t i = 0; i < lhs.size(); i++)
		{
			if (lhs[i] != rhs[i]) return false;
		}
		return true;
	}
}