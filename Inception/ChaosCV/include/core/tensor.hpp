#pragma once

#include "core/def.hpp"
#include "core/types.hpp"
#include "core/array.hpp"
#include "core/buffer.hpp"
#include "core/allocator.hpp"

#include <cstring>
#include <memory>

namespace chaos
{
	enum class Depth
	{
		D1 = 1,
		D2 = 2,
		D4 = 4,
		D8 = 8,
	};

	enum class Packing
	{
		CHW = 1,
		C2HW2 = 2,
		C3HW3 = 3,
		C4HW4 = 4,
		C8HW8 = 8,
	};

	template<class Type>
	inline Type operator*(const Type& val, const Depth& depth)
	{
		return val * static_cast<Type>(depth);
	}

	template<class Type>
	inline Type operator*(const Type& val, const Packing& packing)
	{
		return val * static_cast<Type>(packing);
	}

	// Tensor how ?
	// When Tensor is a scalar dims = 0 <- shape.size() = 1 (definition is 0)
	// When Tensor is a vector dims = 1 <- shape.size() = 1
	// When Tensor is a matrix dims = 2 <- shape.size() = 2
	// When Tensor is a cube   dims = 3 <- shape.size() = 3
	// When Tensor is a N-D    dims = N <- shape.size() = N
	// So the question is that how can I create a Tensor with shape = Shape()
	// 1. Tensor without scalar
	// 2. ????
	class CHAOS_API Tensor
	{
	public:
		Tensor() = default;
		Tensor(const Shape& shape, const Depth& depth = Depth::D1, const Packing& packing = Packing::CHW, Allocator* allocator = nullptr);
		Tensor(const Shape& shape, const Depth& depth, const Packing& packing, void* data, const Steps& steps = Steps());

		template<class Type>
		Tensor(const Array<Type>& arr, Allocator* allocator = nullptr) // copy data from Array
		{
			Create(Shape(static_cast<int>(arr.size())), Steps(1), static_cast<Depth>(sizeof(Type)), Packing::CHW, allocator);
			if (data) memcpy(data, arr.data(), arr.size() * sizeof(Type)); // C6387
		}
		// if constexpr (std::same_as<>Complex, Type)
		//template<>
		Tensor(const Array<Complex>& arr, Allocator* allocator)
		{
			Create(Shape(static_cast<int>(arr.size())), Steps(1), Depth::D4, Packing::C2HW2, allocator);
			if (data) memcpy(data, arr.data(), arr.size() * 8); // C6387
		}

		~Tensor();

		Tensor(const Tensor& tensor);
		Tensor& operator=(const Tensor& tensor);

		void Create(const Shape& new_shape, const Steps& new_steps, const Depth& new_depth, const Packing& new_packing, Allocator* new_allocator = nullptr);

		void CreateLike(const Tensor& tensor, Allocator* allocator = nullptr);

		/// <summary> ref_cnt-- </summary>
		void Release();

		/// <summary> ref_cnt++ </summary>
		void AddRef() noexcept { if (ref_cnt) CHAOS_XADD(ref_cnt, 1); }

		template<class Type = float, Arithmetic<Type> = true, class...Index>
		Type& At(Index&&...idx) const
		{
			Array<int> index = { static_cast<int>(idx)... };
			DCHECK_EQ(shape.size(), index.size()) << "dims expect " << shape.size() << " but got " << index.size();
			size_t offset = 0;
			for (size_t i = 0; i < index.size(); i++)
			{
				DCHECK_LT(index[i], shape[i]) << "out of range at " << i << "th dim";
				offset += steps[i] * index[i];
			}
			return ((Type*)data)[offset];
		}

		float& operator[](size_t idx) const noexcept { return ((float*)data)[idx]; }

		void* data = nullptr;
		Allocator* allocator = nullptr;
		int* ref_cnt = nullptr;

		Shape shape;
		Steps steps;

		Depth depth = Depth::D1;
		Packing packing = Packing::CHW;
	};
}