#pragma once

#include "core/log.hpp"
#include "core/types.hpp"

#define ALIGNMENT 16

// exchange-add operation for atomic operations on reference counters
// Just for windows, reference to NCNN
#ifdef _WIN32
#define CHAOS_XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile*)addr, delta)
#else
#define CHAOS_XADD(addr, delta)
#endif

#ifdef _WIN32
#define ALIGNED_MALLOC(size, alignment) _aligned_malloc(size, alignment)
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
#define ALIGNED_MALLOC(size, alignment) memalign(alignment, size)
#define ALIGNED_FREE(ptr) free(ptr)
#endif

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


	/// <summary>
	/// <para>Aligns a pointer to the specified number of bytes</para>
	/// <para>The function returns the aligned pointer of the same type as the input pointer:</para>
	/// <para>(_Tp*)(((size_t)ptr + n - 1) and -n)</para>
	/// </summary>
	/// <param name="ptr">Aligned pointer</param>
	/// <param name="n">Alignment size that must be a power of two</param>
	/// <return>The aligned pointer of the same type as the input pointer</return>
	template<class Type> static inline Type* AlignPtr(Type* ptr, int n = (int)sizeof(Type))
	{
		//CHECK((n & (n - 1)) == 0) << "n should be a power of 2.";
		return (Type*)(((size_t)ptr + n - 1) & -n);
	}

	/// <summary>
	/// <para>Aligns a buffer size to the specified number of bytes</para>
	/// <para>The function returns the minimum number that is greater than or equal to sz and is divisible by n :</para>
	/// <para>(sz + n - 1) and -n</para>
	/// </summary>
	/// <param name="sz">Buffer size to align</param>
	/// <param name="n">Alignment size that must be a power of two</param>
	/// <return>The minimum number that is greater than or equal to sz and is divisible by n</return>
	static inline size_t AlignSize(size_t sz, int n)
	{
		//CHECK((n & (n - 1)) == 0) << "n should be a power of 2.";
		return (sz + n - 1) & -n;
	}

	static inline void* FastMalloc(size_t capacity)
	{
		return ALIGNED_MALLOC(capacity, ALIGNMENT);
	}
	static inline void FastFree(void* data)
	{
		ALIGNED_FREE(data);
	}

	class CHAOS_API Allocator
	{
	public:
		virtual ~Allocator() = default;
		virtual void* FastMalloc(size_t capacity) = 0;
		virtual void FastFree(void* data) = 0;
	};

	class VulkanDevice;
	class CHAOS_API VulkanAllocator
	{
	public:
		VulkanAllocator(const VulkanDevice* vkdev);
		virtual ~VulkanAllocator();

		virtual VulkanBufferMemory* FastMalloc(size_t capacity) = 0;
		virtual void FastFree(VulkanBufferMemory* data) = 0;

		const VulkanDevice* vkdev;
		bool mappable;
		bool coherent;
		
		void Flush(VulkanBufferMemory* data);
		void Invalidate(VulkanBufferMemory* data);

	protected:
		BufferUsageFlag buffer_usage;
		uint32 memory_type_index = -1;

		VkBuffer CreateBuffer(size_t size);
		VkDeviceMemory AllocateMemory(size_t size, uint32 memory_type_index);
	};
	class CHAOS_API VulkanLocalAllocator : public VulkanAllocator
	{
	public:
		VulkanLocalAllocator(const VulkanDevice* vkdev);
		VulkanBufferMemory* FastMalloc(size_t capacity) override;
		void FastFree(VulkanBufferMemory* data) override;
	};
	class CHAOS_API VulkanStagingAllocator : public VulkanAllocator
	{
	public:
		VulkanStagingAllocator(const VulkanDevice* vkdev);
		VulkanBufferMemory* FastMalloc(size_t capacity) override;
		void FastFree(VulkanBufferMemory* data) override;
	};
}