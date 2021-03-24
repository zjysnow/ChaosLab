#pragma once

#include "core/def.hpp"
#include "core/log.hpp"
#include "core/gpu.hpp"

namespace chaos
{
	/// <summary>
	/// <para>Automatically Allocated Buffer Class</para>
	/// <para>The class is used for temporary buffers in functions and methods.</para>
	/// <para>If a temporary buffer is usually small (a few K's of memory),</para>
	/// <para>but its size depends on the parameters, it makes sense to create a small</para>
	/// <para>fixed-size array on stack and use it if it's large enough. If the required buffer size</para>
	/// <para>is larger than the fixed size, another buffer of sufficient size is allocated dynamically</para>
	/// <para>and released after the processing. Therefore, in typical cases, when the buffer size is small,</para>
	/// <para>there is no overhead associated with malloc()/free().</para>
	/// <para>At the same time, there is no limit on the size of processed data.</para>
	/// <para>This is what AutoBuffer does. The template takes 2 parameters - type of the buffer elements and</para>
	/// <para>the number of stack-allocated elements.</para>
	/// </summary>
	template<class Type, size_t fixed_size = 1024 / sizeof(Type) + 8> class AutoBuffer
	{
	public:
		//! the default constructor
		AutoBuffer()
		{
			ptr = buf;
			sz = fixed_size;
		}

		//! constructor taking the real buffer size
		explicit AutoBuffer(size_t size_)
		{
			ptr = buf;
			sz = fixed_size;
			Allocate(size_);
		}

		//! the copy constructor
		AutoBuffer(const AutoBuffer<Type, fixed_size>& abuf)
		{
			ptr = buf;
			sz = fixed_size;
			Allocate(abuf.size());
			for (size_t i = 0; i < sz; i++)
				ptr[i] = abuf.ptr[i];
		}
		//! the assignment operator
		AutoBuffer<Type, fixed_size>& operator=(const AutoBuffer<Type, fixed_size>& abuf)
		{
			if (this != &abuf)
			{
				Deallocate();
				Allocate(abuf.size());
				for (size_t i = 0; i < sz; i++)
					ptr[i] = abuf.ptr[i];
			}
			return *this;
		}

		//! destructor. calls deallocate()
		virtual ~AutoBuffer() { Deallocate(); }

		//! allocates the new buffer of size _size. if the _size is small enough, stack-allocated buffer is used
		void Allocate(size_t size_)
		{
			if (size_ <= sz)
			{
				sz = size_;
				return;
			}
			Deallocate();
			sz = size_;
			if (size_ > fixed_size)
			{
				ptr = new Type[size_];
			}
		}
		//! deallocates the buffer if it was dynamically allocated
		void Deallocate()
		{
			if (ptr != buf)
			{
				delete[] ptr;
				ptr = buf;
				sz = fixed_size;
			}
		}
		//! resizes the buffer and preserves the content
		void Resize(size_t size_)
		{
			if (size_ <= sz)
			{
				sz = size_;
				return;
			}
			size_t i, prevsize = sz, minsize = std::min(prevsize, size_);
			Type* prevptr = ptr;

			ptr = size_ > fixed_size ? new Type[size_] : buf;
			sz = size_;

			if (ptr != prevptr)
				for (i = 0; i < minsize; i++)
					ptr[i] = prevptr[i];
			for (i = prevsize; i < size_; i++)
				ptr[i] = Type();

			if (prevptr != buf)
				delete[] prevptr;
		}
		//! returns the current buffer size
		size_t size() const noexcept { return sz; }
		//! returns pointer to the real buffer, stack-allocated or heap-allocated
		Type* data() noexcept { return ptr; }
		//! returns read-only pointer to the real buffer, stack-allocated or heap-allocated
		const Type* data() const noexcept { return ptr; }

		//! returns a reference to the element at specified location. No bounds checking is performed in Release builds.
		Type& operator[] (size_t i)
		{
			CHECK_LT(i, sz) << "AutoBuffer out of range. (" << i << " vs " << sz << ")";
			return ptr[i];
		}
		//! returns a reference to the element at specified location. No bounds checking is performed in Release builds.
		const Type& operator[] (size_t i) const
		{
			CHECK_LT(i, sz) << "AutoBuffer out of range. (" << i << " vs " << sz << ")";
			return ptr[i];
		}

	protected:
		//! pointer to the real buffer, can point to buf if the buffer is small enough
		Type* ptr;
		//! size of the real buffer
		size_t sz;
		//! pre-allocated buffer. At least 1 element to confirm C++ standard requirements
		Type buf[(fixed_size > 0) ? fixed_size : 1] = { 0 };
	};

	//class VulkanBuffer
	//{
	//public:
	//	VkBuffer buffer;

	//	size_t capacity;
	//	void* mapped_data;
	//	int ref_cnt;
	//};

	//class VulkanImage
	//{
	//public:
	//	VkImage image;
	//};
}