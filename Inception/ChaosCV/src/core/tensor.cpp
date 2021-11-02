#include "core/tensor.hpp"

namespace chaos
{
	Tensor::Tensor(const Shape& shape, const Depth& depth, const Packing& packing, Allocator* allocator)
	{
		Create(shape, shape.steps(), depth, packing, allocator);
	}
	Tensor::Tensor(const Shape& shape, const Depth& depth, const Packing& packing, void* data, const Steps& steps) :
		data(data), ref_cnt(nullptr), allocator(nullptr),
		shape(shape), steps(steps), depth(depth), packing(packing)
	{
		if (steps.size() == 0) Tensor::steps = shape.steps();
	}

	Tensor::~Tensor()
	{
		Release();
	}

	Tensor::Tensor(const Tensor& tensor) :
		data(tensor.data), ref_cnt(tensor.ref_cnt), allocator(tensor.allocator),
		shape(tensor.shape), steps(tensor.steps), depth(tensor.depth), packing(tensor.packing)
	{
		if (ref_cnt) CHAOS_XADD(ref_cnt, 1);
	}
	Tensor& Tensor::operator=(const Tensor& tensor)
	{
		if (this == &tensor) return *this;

		if (tensor.ref_cnt) CHAOS_XADD(tensor.ref_cnt, 1);

		Release();

		data = tensor.data;
		ref_cnt = tensor.ref_cnt;
		allocator = tensor.allocator;

		shape = tensor.shape;
		steps = tensor.steps;
		depth = tensor.depth;
		packing = tensor.packing;

		return *this;
	}

	void Tensor::Create(const Shape& new_shape, const Steps& new_steps, const Depth& new_depth, const Packing& new_packing, Allocator* new_allocator)
	{
		if (shape == new_shape && steps == new_steps && depth == new_depth && packing == new_packing && allocator == new_allocator)
			return;

		Release();

		allocator = new_allocator;

		shape = new_shape;
		steps = new_steps;
		depth = new_depth;
		packing = new_packing;

		size_t total = static_cast<size_t>(shape[0]) * steps[0];
		if (total > 0)
		{
			size_t capacity = AlignSize(total * depth * packing, 4);
			if (allocator)
			{
				data = allocator->FastMalloc(capacity + sizeof(ref_cnt));
			}
			else
			{
				data = FastMalloc(capacity + sizeof(ref_cnt));
			}
			ref_cnt = (int*)((uchar*)data + capacity);
			*ref_cnt = 1;
		}
	}

	void Tensor::CreateLike(const Tensor& tensor, Allocator* allocator)
	{
		Create(tensor.shape, tensor.steps, tensor.depth, tensor.packing, allocator);
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
		allocator = nullptr;
	}

	void Tensor::CopyTo(Tensor& tensor) const
	{
		if (this == &tensor) return;
		DCHECK(not tensor.empty());
		DCHECK_EQ(shape, tensor.shape) << "expect " << shape << " but got " << tensor.shape;
		if (steps == tensor.steps)
		{
			memcpy(tensor.data, data, total() * depth * packing);
		}
		else
		{
			// 'row' is always contiguous
			size_t dims = shape.size();
			size_t esize = 1 * depth * packing;
			int rsize = shape[dims - 1];
			for (size_t i = 0; i < shape.total(); i += rsize)
			{
				size_t dofst = 0;
				size_t sofst = 0;
				size_t idx = i;
				for (size_t d = 0; d < dims; d++)
				{
					size_t k = idx % shape[dims - d - 1];
					dofst += k * tensor.steps[dims - d - 1];
					sofst += k * steps[dims - d - 1];
					idx /= shape[dims - d - 1];
				}
				memcpy((uchar*)tensor.data + dofst * esize, (const uchar*)data + sofst * esize, rsize * esize);
			}
		}
	}

	Tensor Tensor::Clone(Allocator* allocator) const
	{
		Tensor tensor = Tensor(shape, depth, packing, allocator);
		CopyTo(tensor);
		return tensor;
	}
}