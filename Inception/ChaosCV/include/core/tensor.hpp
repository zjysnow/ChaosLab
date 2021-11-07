#pragma once

#include "core/def.hpp"
#include "core/log.hpp"
#include "core/types.hpp"
#include "core/array.hpp"
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
		Tensor(const Array<Complex>& arr, Allocator* allocator = nullptr)
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

		void CopyTo(Tensor& tensor) const;
		Tensor Clone(Allocator* allocator = nullptr) const;

		// the new shape should be the subsapce of the Tensor shape
		// example:
		// shape = [3,4,5,6]
		// new_shape can be [5,1] for the `at`-th col data
		// or be [5,6] for the `at`-th channel data
		// means that new_shape should broadcast to the Tensor shape
		// this wouldn't copy any data from the source Tensor
		Tensor Cut(const Shape& new_shape, int at) const;

		/// <summary> ref_cnt++ </summary>
		void AddRef() noexcept { if (ref_cnt) CHAOS_XADD(ref_cnt, 1); }

		template<class Type = float, class...Index, Arithmetic<Type> = true>
		Type& At(Index&&...idx) const
		{
			Array<unsigned int> index = { static_cast<unsigned int>(idx)... };
			DCHECK_EQ(shape.size(), index.size()) << "dims expect " << shape.size() << " but got " << index.size();
			size_t offset = 0;
			for (size_t i = 0; i < index.size(); i++)
			{
				//DCHECK_GE(index[i], 0) << "expect index[" << i << "] >= 0, but got " << index[i];
				DCHECK_LT(index[i], shape[i]) << "expect index[" << i << "] < " << shape[i] << " but got " << index[i];
				offset += steps[i] * index[i];
			}
			return ((Type*)data)[offset];
		}

		bool empty() const noexcept { return shape.size() == 0 || data == nullptr; }
		bool contiguous() const noexcept { return empty() ? true : (shape.total() == shape[0] * steps[0]); }
		size_t total() const noexcept { return empty() ? 0 : static_cast<size_t>(shape[0]) * steps[0]; }
		
		Tensor row(int at) const;
		Tensor col(int at) const;
		Tensor channel(int at) const;

		float& operator[](size_t idx) const noexcept { return ((float*)data)[idx]; }

		static Tensor randn(const Shape& shape, float mu = 0.f, float sigma = 1.f, Allocator* allocator = nullptr);
		static Tensor randu(const Shape& shape, float min = 0.f, float max = 1.f, Allocator* allocator = nullptr);
		static Tensor zeros(const Shape& shape, Allocator* allocator = nullptr);
		static Tensor eye(int h, int w, Allocator* allocator = nullptr);

		void* data = nullptr;
		Allocator* allocator = nullptr;
		int* ref_cnt = nullptr;

		Shape shape;
		Steps steps;

		Depth depth = Depth::D1;
		Packing packing = Packing::CHW;
	};
}
