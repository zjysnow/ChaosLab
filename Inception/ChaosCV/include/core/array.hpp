#pragma once

#include "core/def.hpp"
#include "core/types.hpp"

#include <memory>
#include <type_traits>

namespace chaos
{
	template<class Type>
	concept Arithmetic = std::integral<Type> or std::floating_point<Type> or std::same_as<Complex, Type>;

	template<Arithmetic Type>
	class Array
	{
	public:
		Array() = default;
		Array(size_t count)
		{
			Create(count);
		}
		Array(const Type& val, size_t count)
		{
			Create(count, &val);
		}
		Array(const std::initializer_list<Type>& list)
		{
			Create(list.size(), list.begin(), 1);
		}

		virtual ~Array() { Release(); }

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
						data_[i] = arr[i];
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

		Array(Array&& arr) noexcept
		{
			data_ = arr.data_;
			size_ = arr.size_;
			arr.data_ = nullptr;
			arr.size_ = 0;
		}
		Array& operator=(Array&& arr) noexcept
		{
			Release(); // clear this and steal from arr
			data_ = arr.data_;
			size_ = arr.size_;
			arr.data_ = nullptr;
			arr.size_ = 0;
			return *this;
		}

		Type& operator[](size_t off) noexcept { return data_[off]; }
		const Type& operator[](size_t off) const noexcept { return data_[off]; }

		size_t size() const noexcept { return size_; }
		Type* data() const noexcept { return data_; }

		void Resize(size_t new_size)
		{
			Release();
			Create(new_size);
		}
		void Resize(size_t new_size, const Type& val)
		{
			Release();
			Create(new_size, &val);
		}
	protected:
		Type* data_ = nullptr;
		size_t size_ = 0;

	private:
		void Create(size_t new_size)
		{
			size_ = new_size;
			if (size_ > 0)
			{
				//size_t capacity = AlignSize(size_, alignof(Type));
				//data = static_cast<Type*>(FastMalloc(capacity));
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
				//FastFree(data);
				for (size_t i = 0; i < size_; i++)
				{
					data_[i].~Type();
				}
				::operator delete (static_cast<void*>(data_), std::align_val_t{ alignof(Type) });
			}
			data_ = nullptr;
			size_ = 0;
			
		}
	};

	template <class Type>
	Type* begin(Array<Type>& arr)
	{
		return &arr[0];
	}

	template <class Type>
	const Type* begin(const Array<Type>& arr)
	{
		return &arr[0];
	}

	template <class Type>
	Type* end(Array<Type>& arr)
	{
		return &arr[0] + arr.size();
	}

	template <class Type>
	const Type* end(const Array<Type>& arr)
	{
		return &arr[0] + arr.size();
	}

	class CHAOS_API Steps : public Array<uint32>
	{
	public:
		friend class Shape;

		Steps() = default;
		Steps(int s0);
		Steps(int s0, int s1);

		~Steps() = default;

		void Expand(size_t axis, int dims, uint32 step);
		bool operator==(const Steps& rhs);
	};

	class CHAOS_API Shape : public Array<uint32>
	{
	public:
		Shape() = default;
		Shape(int d0);
		Shape(int d0, int d1);
		Shape(int d0, int d1, int d2);

		template<class Type> requires std::is_convertible_v<Type, uint32>
		Shape(const std::initializer_list<Type>& list) : Array<uint32>(list.size())
		{
			for (size_t idx = 0; const auto & data : list)
			{
				data_[idx++] = static_cast<uint32>(data);
			}
		}

		Shape(const Array<uint32>& arr) : Array<uint32>(arr) {}
		~Shape() = default;

		void Expand(size_t axis, int dims = 1);

		uint32 total() const;
		Steps steps() const;

		bool operator==(const Shape& rhs);
		CHAOS_API friend std::ostream& operator<<(std::ostream& stream, const Shape& shape);
	};
}