#include "core/tensor.hpp"
#include "core/buffer.hpp"

namespace chaos
{
	Tensor::Tensor(const Shape& shape, const DataType& dtype, const Packing& packing, Allocator* allocator)
	{
		Create(shape, shape.steps(), dtype, packing, allocator);
	}

	Tensor::Tensor(const Shape& shape_, const DataType& dtype_, const Packing& packing_, void* data_, const Steps& steps_)
	{
		data = data_;

		shape = shape_;
		dtype = dtype_;
		packing = packing_;

		steps = 0 == steps_.size ? shape.steps() : steps_;
	}

	Tensor::Tensor(const Tensor& tensor)
	{
		data = tensor.data;
		ref_cnt = tensor.ref_cnt;
		allocator = tensor.allocator;

		shape = tensor.shape;
		steps = tensor.steps;
		dtype = tensor.dtype;
		packing = tensor.packing;

		if (ref_cnt) CHAOS_XADD(ref_cnt, 1);
	}

	Tensor& Tensor::operator=(const Tensor& tensor)
	{
		if (this == &tensor) return *this;

		if (tensor.ref_cnt) CHAOS_XADD(tensor.ref_cnt, 1);

		data = tensor.data;
		ref_cnt = tensor.ref_cnt;
		allocator = tensor.allocator;

		shape = tensor.shape;
		steps = tensor.steps;
		dtype = tensor.dtype;
		packing = tensor.packing;

		return *this;
	}

	void Tensor::Create(const Shape& shape_, const Steps& steps_, const DataType& dtype_, const Packing& packing_, Allocator* allocator_)
	{
		if (shape_ == shape && steps_ == steps && dtype_ == dtype && packing_ == packing_ && allocator_ == allocator) return;

		Release();

		shape = shape_;
		steps = steps_;
		dtype = dtype_;
		packing = packing_;
		allocator = allocator_;

		size_t total = (size_t)shape[0] * steps[0];
		if (total > 0)
		{
			size_t capacity = AlignSize(total * dtype * packing, 4);

			if (allocator)
			{
				data = allocator->FastaMalloc(capacity + sizeof(*ref_cnt));
			}
			else
			{
				data = FastMalloc(capacity + sizeof(*ref_cnt));
			}
			ref_cnt = (int*)((uchar*)data + capacity);
			*ref_cnt = 1;
		}
	}
	void Tensor::CreateLike(const Tensor& tensor, Allocator* allocator)
	{
		Create(tensor.shape, tensor.steps, tensor.dtype, tensor.packing, allocator);
	}

	void Tensor::Release()
	{
		if (ref_cnt && CHAOS_XADD(ref_cnt, -1) == 1)
		{
			if (allocator)
			{
				allocator->FastFree(data);
			}
			else
			{
				FastFree(data);
			}
		}

		data = nullptr;
		ref_cnt = nullptr;

		dtype = DataType::D1;
		packing = Packing::CHW;

		shape = Shape();
		steps = Steps();
	}

	void Tensor::CopyTo(Tensor& tensor) const
	{
		if (this == &tensor) return;
		if (tensor.empty()) tensor.CreateLike(*this);

		CHECK_EQ(shape, tensor.shape) << "expect " << shape << " but got " << tensor.shape;

		if (steps == tensor.steps)
		{
			size_t capacity = AlignSize((size_t)shape[0] * steps[0] * dtype * packing, 4);
			memcpy(tensor.data, data, capacity);
		}
		else // steps != tensor.steps
		{
			size_t dims = shape.dims;
			size_t esize = 1ULL * dtype * packing;
			size_t rsize = shape[dims - 1];
			size_t rows = shape.total() / rsize;
			for (size_t r = 0; r < rows; r++)
			{
				size_t dst_offset = 0;
				size_t src_offset = 0;
				size_t idx = r * rsize;
				for (size_t d = 0; d < dims; d++)
				{
					size_t k = idx % shape[dims - d - 1];
					dst_offset += k * tensor.steps[dims - d  -1];
					src_offset += k * steps[dims - d - 1];
					idx /= shape[dims - d - 1];
				}
				memcpy((uchar*)tensor.data + dst_offset * esize, (uchar*)data + src_offset * esize, rsize * esize);
			}
		}
	}
	Tensor Tensor::Clone(Allocator* allocator) const
	{
		Tensor tensor = Tensor(shape, dtype, packing, allocator);
		CopyTo(tensor);
		return tensor;
	}



	VulkanTensor::VulkanTensor(const Shape& shape, const DataType& dtype, const Packing& packing, VulkanAllocator* allocator)
	{
		Create(shape, shape.steps(), dtype, packing, allocator);
	}
	void VulkanTensor::Create(const Shape& shape_, const Steps& steps_, const DataType& dtype_, const Packing& packing_, VulkanAllocator* allocator_)
	{
		if (shape_ == shape && steps_ == steps && dtype_ == dtype && packing_ == packing && allocator_ == allocator) return;

		Release();

		shape = shape_;
		steps = steps_;
		dtype = dtype_;
		packing = packing_;
		allocator = allocator_;

		size_t total = (size_t)shape[0] * steps[0];
		if (total > 0)
		{
			size_t capacity = AlignSize(total * dtype * packing, 4);

			data = allocator->FastMalloc(capacity);

			ref_cnt = (int*)((uchar*)data + offsetof(VulkanBuffer, ref_cnt));
			*ref_cnt = 1;
		}
	}
	void VulkanTensor::CreateLike(const Tensor& tensor, VulkanAllocator* allocator)
	{
		Create(tensor.shape, tensor.steps, tensor.dtype, tensor.packing, allocator);
	}
	void VulkanTensor::CreateLike(const VulkanTensor& tensor, VulkanAllocator* allocator)
	{
		Create(tensor.shape, tensor.steps, tensor.dtype, tensor.packing, allocator);
	}

	void VulkanTensor::Release()
	{
		if (ref_cnt && CHAOS_XADD(ref_cnt, -1) == 1)
		{
			if (allocator && data)
			{
				allocator->FastFree(data);
			}
		}

		data = nullptr;
		ref_cnt = nullptr;

		shape = Shape();
		steps = Steps();
	}

	Tensor VulkanTensor::Mapped() const
	{
		return Tensor(shape, dtype, packing, mapped_data(), steps);
	}

	void* VulkanTensor::mapped_data() const
	{
		if (not allocator->mappable) return nullptr;
		return (uchar*)data->mapped_data + data->offset;
	}
}