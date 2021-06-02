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