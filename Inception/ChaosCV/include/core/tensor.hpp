#pragma once

#include "core/core.hpp"
#include "core/array.hpp"
#include "core/vulkan.hpp"
#include "core/allocator.hpp"

namespace chaos
{
	class VulkanTensor;
	class CHAOS_API Tensor
	{
	public:
		Tensor() = default;
		Tensor(const Shape& shape, const Depth& depth, const Packing& packing, Allocator* allocator = nullptr);
		Tensor(const Shape& shape, const Depth& depth, const Packing& packing, void* data, const Steps& steps = Steps());

		template<class Type>
		Tensor(const Array<Type>& arr, Allocator* allocator = nullptr)
		{
			if constexpr (std::same_as<Complex, Type>)
			{
				Create(Shape((int)arr.size()), Steps(1), Depth::D4, Packing::C2HW2, allocator);
				float* data_ = static_cast<float*>(data);
				for (size_t i = 0; const auto& val : arr)
				{
					data_[i++] = val.re;
					data_[i++] = val.im;
				}
			}
			else
			{
				Create(Shape((int)arr.size()), Steps(1), static_cast<Depth>(sizeof(Type)), Packing::CHW, allocator);
				Type* data_ = static_cast<Type*>(data);
				for (size_t i = 0; const auto & val : arr)
				{
					data_[i++] = val;
				}
			}
		}

		template<Arithmetic Type>
		Tensor(const std::initializer_list<Type> list, Allocator* allocator = nullptr)
		{
			if constexpr (std::same_as<Complex, Type>)
			{
				Create(Shape((int)list.size()), Steps(1), Depth::D4, Packing::C2HW2, allocator);
				float* data_ = static_cast<float*>(data);
				for (size_t i = 0; const auto & val : list)
				{
					data_[i++] = val.re;
					data_[i++] = val.im;
				}
			}
			else
			{
				Create(Shape((int)list.size()), Steps(1), static_cast<Depth>(sizeof(Type)), Packing::CHW, allocator);
				Type* data_ = static_cast<Type*>(data);
				for (size_t i = 0; const auto & val : list)
				{
					data_[i++] = val;
				}
			}
		}

		~Tensor();

		Tensor(const Tensor& tensor);
		Tensor& operator=(const Tensor& tensor);

		void Create(const Shape& shape, const Steps& steps, const Depth& depth, const Packing& packing, Allocator* allocator);
		/// <summary> ref_cnt-- </summary>
		void Release();

		void CreateLkie(const Tensor& tensor, Allocator* allocator = nullptr);
		void CreateLike(const VulkanTensor& tensor, Allocator* allocator = nullptr);

		/// <summary> ref_cnt++ </summary>
		void AddRef() noexcept { if (ref_cnt) CHAOS_XADD(ref_cnt, 1); }

		void CopyTo(Tensor& tensor) const;
		Tensor Clone(Allocator* allocator = nullptr) const;

		uint32 total() const noexcept { return shape[0] * steps[0]; }
		bool empty() const noexcept { return shape.size() == 0 || data == nullptr; }

		float& operator[](size_t idx) noexcept { return ((float*)data)[idx]; }
		const float& operator[](size_t idx) const noexcept { return ((float*)data)[idx]; }

		template<class Type = float, class ...Index> 
		requires (std::is_arithmetic_v<Type>) const Type& At(Index... idx) const
		{
			Array<uint32> index = { static_cast<uint32>(idx)... };
			CHECK_EQ(shape.size(), index.size()) << "dims expect " << shape.size() << " but got " << index.size();
			size_t offset = 0.f;
			for (size_t i = 0; i < index.size(); i++)
			{
				CHECK_LT(index[i], shape[i]) << "out of range.";
				offset += steps[i] * index[i];
			}
			return ((Type*)data)[offset];
		}
		
		template<class Type = float, class ...Index>
		requires (std::is_arithmetic_v<Type>) Type& At(Index... idx)
		{
			Array<uint32> index = { static_cast<uint32>(idx)... };
			CHECK_EQ(shape.size(), index.size()) << "dims expect " << shape.size() << " but got " << index.size();
			size_t offset = 0.f;
			for (size_t i = 0; i < index.size(); i++)
			{
				CHECK_LT(index[i], shape[i]) << "out of range.";
				offset += steps[i] * index[i];
			}
			return ((Type*)data)[offset];
		}

		static Tensor randn(const Shape& shape, float mu = 0.f, float sigma = 1.f, Allocator* allocator = nullptr);

		void* data = nullptr;
		Allocator* allocator = nullptr;
		int* ref_cnt = nullptr;

		Depth depth = Depth::D1;
		Packing packing = Packing::CHW;
		Shape shape;
		Steps steps;
	};
	

	class CHAOS_API VulkanTensor
	{
	public:
		VulkanTensor() = default;
		VulkanTensor(const Shape& shape, const Depth& depth, const Packing& packing, VulkanAllocator* allocator);
		~VulkanTensor();

		VulkanTensor(const VulkanTensor& tensor);
		VulkanTensor& operator=(const VulkanTensor& tensor);

		void Create(const Shape& shape, const Steps& steps, const Depth& depth, const Packing& packing, VulkanAllocator* allocator);
		void Release();

		void CreateLike(const Tensor& tensor, VulkanAllocator* allocator);
		void CreateLike(const VulkanTensor& tensor, VulkanAllocator* allocator);

		/// <summary> ref_cnt++ </summary>
		void AddRef() noexcept { if (ref_cnt) CHAOS_XADD(ref_cnt, 1); }

		uint32 total() const noexcept { return shape[0] * steps[0]; }

		bool empty() const noexcept { return data == nullptr || shape.size() == 0; }
		// low-level reference
		void* mapped_data() const noexcept { CHECK(allocator->mappable);  return (uchar*)data->mapped_data + data->offset; }
		VkBuffer buffer() const noexcept { return data->buffer; }
		size_t buffer_offset() const noexcept { return data->offset; }
		size_t buffer_capacity() const noexcept { return data->capacity; }

		VulkanBufferMemory* data = nullptr;
		VulkanAllocator* allocator = nullptr;
		int* ref_cnt = nullptr;

		Depth depth = Depth::D1;
		Packing packing = Packing::CHW;
		Shape shape;
		Steps steps;
	};
}