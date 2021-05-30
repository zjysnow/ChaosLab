#include "core/tensor.hpp"

#include <vulkan/vulkan.hpp>

namespace chaos
{
	Tensor::Tensor(const Shape& shape, const Depth& depth, const Packing& packing, Allocator* allocator)
	{
		Create(shape, shape.steps(), depth, packing, allocator);
	}
	Tensor::Tensor(const Shape& shape_, const Depth& depth_, const Packing& packing_, void* data_, const Steps& steps_)
	{
		data = data_;

		shape = shape_;
		depth = depth_;
		packing = packing_;

		steps = 0 == steps_.size() ? shape.steps() : steps_;
	}
	Tensor::~Tensor() { Release(); }

	Tensor::Tensor(const Tensor& tensor)
	{
		data = tensor.data;
		ref_cnt = tensor.ref_cnt;
		allocator = tensor.allocator;

		shape = tensor.shape;
		steps = tensor.steps;
		depth = tensor.depth;
		packing = tensor.packing;

		if (ref_cnt) CHAOS_XADD(ref_cnt, 1);
	}
	Tensor& Tensor::operator=(const Tensor& tensor)
	{
		if (this != &tensor)
		{
			if (tensor.ref_cnt) CHAOS_XADD(tensor.ref_cnt, 1);

			data = tensor.data;
			ref_cnt = tensor.ref_cnt;
			allocator = tensor.allocator;

			shape = tensor.shape;
			steps = tensor.steps;
			depth = tensor.depth;
			packing = tensor.packing;
		}
		return *this;
	}

	void Tensor::Create(const Shape& shape_, const Steps& steps_, const Depth& depth_, const Packing& packing_, Allocator* allocator_)
	{
		if (shape_ == shape && steps_ == steps && depth_ == depth && packing_ == packing && allocator_ == allocator) return;

		Release();

		shape = shape_;
		steps = steps_;
		depth = depth_;
		packing = packing_;
		allocator = allocator_;

		uint32 size = shape[0] * steps[0];
		if (size > 0)
		{
			size_t capacity = AlignSize(size * depth * packing, 4);

			if (allocator)
			{
				data = allocator->FastMalloc(capacity + sizeof(*ref_cnt));
			}
			else
			{
				data = FastMalloc(capacity + sizeof(*ref_cnt));
			}
			ref_cnt = (int*)((uchar*)data + capacity);
			*ref_cnt = 1;
		}
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
		depth = Depth::D1;
		packing = Packing::CHW;
		shape = Shape();
		steps = Steps();
	}

	void Tensor::CreateLkie(const Tensor& tensor, Allocator* allocator)
	{
		Create(tensor.shape, tensor.steps, tensor.depth, tensor.packing, allocator);
	}
	void Tensor::CreateLike(const VulkanTensor& tensor, Allocator* allocator)
	{
		Create(tensor.shape, tensor.steps, tensor.depth, tensor.packing, allocator);
	}







	VulkanTensor::VulkanTensor(const Shape& shape, const Depth& depth, const Packing& packing, VulkanAllocator* allocator)
	{
		Create(shape, shape.steps(), depth, packing, allocator);
	}
	VulkanTensor::~VulkanTensor() { Release(); }
	VulkanTensor::VulkanTensor(const VulkanTensor& tensor)
	{
		data = tensor.data;
		ref_cnt = tensor.ref_cnt;
		allocator = tensor.allocator;

		shape = tensor.shape;
		steps = tensor.steps;
		depth = tensor.depth;
		packing = tensor.packing;

		if (ref_cnt) CHAOS_XADD(ref_cnt, 1);
	}
	VulkanTensor& VulkanTensor::operator=(const VulkanTensor& tensor)
	{
		if (this != &tensor)
		{
			if (tensor.ref_cnt) CHAOS_XADD(tensor.ref_cnt, 1);

			data = tensor.data;
			ref_cnt = tensor.ref_cnt;
			allocator = tensor.allocator;

			shape = tensor.shape;
			steps = tensor.steps;
			depth = tensor.depth;
			packing = tensor.packing;
		}
		return *this;
	}

	void VulkanTensor::Create(const Shape& shape_, const Steps& steps_, const Depth& depth_, const Packing& packing_, VulkanAllocator* allocator_)
	{
		if (shape_ == shape && steps_ == steps && depth_ == depth && packing_ == packing && allocator_ == allocator) return;

		Release();

		shape = shape_;
		steps = steps_;
		depth = depth_;
		packing = packing_;
		allocator = allocator_;

		uint32 size = shape[0] * steps[0];
		if (size > 0)
		{
			size_t capacity = AlignSize(size * depth * packing, 4);
			data = allocator->FastMalloc(capacity);

			ref_cnt = (int*)((uchar*)data + offsetof(VulkanBufferMemory, ref_cnt));
			*ref_cnt = 1;
		}
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

		depth = Depth::D1;
		packing = Packing::CHW;
		shape = Shape();
		steps = Steps();
	}
	void VulkanTensor::CreateLike(const Tensor& tensor, VulkanAllocator* allocator)
	{
		Create(tensor.shape, tensor.steps, tensor.depth, tensor.packing, allocator);
	}
	void VulkanTensor::CreateLike(const VulkanTensor& tensor, VulkanAllocator* allocator)
	{
		Create(tensor.shape, tensor.steps, tensor.depth, tensor.packing, allocator);
	}
}