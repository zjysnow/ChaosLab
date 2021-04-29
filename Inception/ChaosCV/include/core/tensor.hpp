#pragma once

#include "core/core.hpp"

namespace chaos
{
	class VulkanBuffer;
	class VulkanAllocator;
	class VulkanTensor;
	class CHAOS_API Tensor
	{
	public:
		Tensor() = default;
		~Tensor() { Release(); }

		Tensor(const Shape& shape, const DataType& dtype = DataType::D4, const Packing& packing = Packing::CHW, Allocator* allocator = nullptr);
		Tensor(const Shape& shape, const DataType& dtype, const Packing& packing, void* data, const Steps& steps = Steps());

		Tensor(const Tensor& tensor);
		Tensor& operator=(const Tensor& tensor);

		template<class Type, std::enable_if_t<std::is_arithmetic_v<Type>, bool> = true>
		Tensor(const std::initializer_list<Type>& vec)
		{
			uint32 sz = (uint32)vec.size();
			Create({ sz }, { 1 }, static_cast<DataType>(sizeof(Type)), Packing::CHW, nullptr);
			memcpy(data, vec.begin(), sz * sizeof(Type));
		}

		void Create(const Shape& shape, const Steps& steps, const DataType& dtype, const Packing& packing, Allocator* allocator);
		void CreateLike(const Tensor& tensor, Allocator* allocator = nullptr);

		/// <summary> ref_cnt-- </summary>
		void Release();

		void CopyTo(Tensor& tensor) const;
		Tensor Clone(Allocator* allocator = nullptr) const;

		/// <summary> ref_cnt++ </summary>
		void AddRef() noexcept { if (ref_cnt) CHAOS_XADD(ref_cnt, 1); }

		bool empty() const noexcept { return data == nullptr || 0 == shape.dims; }

		size_t total() const noexcept { return (size_t)shape[0] * steps[0]; }

		float& operator[](size_t idx) noexcept { return ((float*)data)[idx]; }
		const float& operator[](size_t idx) const noexcept { return ((float*)data)[idx]; }


		template<class Type = float, std::enable_if_t<std::is_arithmetic_v<Type>, bool> = true>
		static Tensor eye(uint32 w, uint32 h, Allocator* allocator = nullptr)
		{
			Tensor eye_;
			eye_.Create(Shape(h, w), /*steps=*/{ w, 1u }, static_cast<DataType>(sizeof(Type)), Packing::CHW, allocator);
			memset(eye_.data, 0, h * w * sizeof(Type));
			Type* data = (Type*)eye_.data;
			uint32 rows = w < h ? w : h; //eye_.shape[0];
			uint32 rstep = w; //eye_.steps[0];
			for (size_t r = 0; r < rows; r++)
			{
				data[r * rstep + r] = 1;
			}
			return eye_;
		}

		template<class Type = float, std::enable_if_t<std::is_arithmetic_v<Type>, bool> = true>
		static Tensor zeros(const Shape& shape, Allocator* allocator = nullptr)
		{
			Tensor zeros_;
			zeros_.Create(shape, shape.steps(), static_cast<DataType>(sizeof(Type)), Packing::CHW, allocator);
			memset(zeros_.data, 0, shape.total() * sizeof(Type));
			return zeros_;
		}

		bool is_continuous() const noexcept;

		void* data = nullptr;
		Allocator* allocator = nullptr;
		int* ref_cnt = nullptr;

		Shape shape;
		Steps steps;

		DataType dtype = DataType::D1;
		Packing packing = Packing::CHW;
	};

	class CHAOS_API VulkanTensor
	{
	public:
		VulkanTensor() = default;
		~VulkanTensor() { Release(); }

		VulkanTensor(const Shape& shape, const DataType& dtype, const Packing& packing, VulkanAllocator* allocator);

		void Create(const Shape& shape, const Steps& steps, const DataType& dtype, const Packing& packing, VulkanAllocator* allocator);
		void CreateLike(const Tensor&, VulkanAllocator* allocator);
		void CreateLike(const VulkanTensor&, VulkanAllocator* allocator);

		void Release();

		Tensor Mapped() const;
		void* mapped_data() const;

		/// <summary> ref_cnt++ </summary>
		void AddRef() noexcept { if (ref_cnt) CHAOS_XADD(ref_cnt, 1); }

		bool empty() const noexcept { return data == nullptr || 0 == shape.dims; }

		VulkanBuffer* data = nullptr;
		VulkanAllocator* allocator = nullptr;
		int* ref_cnt = nullptr;

		Shape shape;
		Steps steps;

		DataType dtype = DataType::D1;
		Packing packing = Packing::CHW;
	};
}