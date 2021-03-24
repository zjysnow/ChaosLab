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

		Tensor(const Shape& shape, const DataType& dtype, const Packing& packing, Allocator* allocator = nullptr);
		Tensor(const Shape& shape, const DataType& dtype, const Packing& packing, void* data, const Steps& steps = Steps());

		Tensor(const Tensor& tensor);
		Tensor& operator=(const Tensor& tensor);

		void Create(const Shape& shape, const Steps& steps, const DataType& dtype, const Packing& packing, Allocator* allocator);
		void CreateLike(const Tensor& tensor, Allocator* allocator = nullptr);

		/// <summary> ref_cnt-- </summary>
		void Release();

		void CopyTo(Tensor& tensor) const;
		Tensor Clone(Allocator* allocator = nullptr) const;

		/// <summary> ref_cnt++ </summary>
		void AddRef() noexcept { if (ref_cnt) CHAOS_XADD(ref_cnt, 1); }

		bool empty() const noexcept { return data == nullptr || 0 == shape.dims; }

		float& operator[](size_t idx) noexcept { return ((float*)data)[idx]; }
		const float& operator[](size_t idx) const noexcept { return ((float*)data)[idx]; }

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

		~VulkanTensor();
		void Release();

		/// <summary> ref_cnt++ </summary>
		void AddRef() noexcept { if (ref_cnt) CHAOS_XADD(ref_cnt, 1); }

		bool empty() const noexcept { return buffer == nullptr || 0 == shape.dims; }

		VulkanBuffer* buffer = nullptr;
		VulkanAllocator* allocator = nullptr;
		int* ref_cnt = nullptr;

		Shape shape;
		Steps steps;

		DataType dtype = DataType::D1;
		Packing packing = Packing::CHW;
	};


	class CHAOS_API InputTensor
	{
	public:
		enum KindFlag
		{
			NONE = 0,
			TENSOR = 1,
			VECTOR_TENSOR = 2,
		};

		InputTensor();
		virtual ~InputTensor();

		InputTensor(const Tensor& t);
		InputTensor(const std::vector<Tensor>& vt);

		

		Tensor GetTensor() const;
		void GetVectorTensor(std::vector<Tensor>& vt) const;

		void* GetObject() const;

		
	protected:
		void Init(int flags, const void* obj);

		int flags;
		void* obj;
	};

	class CHAOS_API OutputTensor : public InputTensor
	{
	public:

	};

	class CHAOS_API InputOutputTensor : public OutputTensor
	{
	public:

	};
}