#pragma once

#include "core/def.hpp"
#include "core/types.hpp"
#include "core/allocator.hpp"

#include <memory>
#include <type_traits>

namespace chaos
{
	template<class Type>
	using Arithmetic = std::enable_if_t<std::is_integral_v<Type> or std::is_floating_point_v<Type> or std::is_same_v<Complex, Type>, bool>;

	template<class Type, Arithmetic<Type> = true>
	class Array
	{
	public:
		Array() = default;
		Array(size_t new_size) { Create(new_size); }
		Array(size_t new_size, Type val) { Create(new_size, &val); }

		Array(const std::initializer_list<Type>& list)
		{
			Create(list.size(), list.begin(), 1);
		}

		virtual ~Array() { Release(); }

		// Copy Constructor
		Array(const Array& arr)
		{
			Create(arr.size_, arr.data_, 1);
		}
		Array& operator=(const Array& arr)
		{
			if (this != std::addressof(arr))
			{
				if (size_ == arr.size_)
				{
					for (size_t i = 0; i < size_; i++)
					{
						data_[i] = arr.data_[i];
					}
				}
				else
				{
					Release();
					Create(arr.size_, arr.data_, 1);
				}
			}
			return *this;
		}

		// Move Constructor
		Array(Array&& arr) noexcept
		{
			std::swap(data_, arr.data_);
			std::swap(size_, arr.size_);
		}
		Array& operator=(Array&& arr) noexcept
		{
			Release(); // clear this and steal from arr
			std::swap(data_, arr.data_);
			std::swap(size_, arr.size_);
			return *this;
		}

		const Type& operator[](int idx) const noexcept { return data_[idx]; }
		Type& operator[](int idx) noexcept { return data_[idx]; }

		size_t size() const noexcept { return size_; }
		const Type* data() const noexcept { return data_; }

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
		//Steps();
	};

	class CHAOS_API Shape : public Array<int>
	{
	public:
		Shape();
		Shape(int d0);
		Shape(int d1, int d0);
		Shape(int d2, int d1, int d0);
	};
}