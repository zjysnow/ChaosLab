#pragma once

#include "core/def.hpp"

#include <malloc.h>

#define ALIGNMENT 16

// exchange-add operation for atomic operations on reference counters
// Just for windows, reference to ncnn
#ifdef _WIN32
#define CHAOS_XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile*)addr, delta)
#else // Linux gcc >= 4.7
#if defined __ATOMIC_ACQ_REL && !defined __clang__
#define CHAOS_XADD(addr, delta) (int)__atomic_fetch_add((unsigned*)(addr), (unsigned)(delta), __ATOMIC_ACQ_REL)
#else
#define CHAOS_XADD(addr, delta) (int)__sync_fetch_and_add((unsigned*)(addr), (unsigned)(delta))
#endif
#endif

#ifdef _WIN32
#define ALIGNED_MALLOC(size, alignment) _aligned_malloc(size, alignment)
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
#define ALIGNED_MALLOC(size, alignment) memalign(alignment, size)
#define ALIGNED_FREE(ptr) free(ptr)
#endif

#ifndef _WIN32
namespace std
{
	template<class T, class...Args>
	constexpr T* construct_at(T* p, Args&&...args)
	{
		return ::new (const_cast<void*>(static_cast<const volatile void*>(p)))
			T(std::forward<Args>(args)...);
	}
}
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
}