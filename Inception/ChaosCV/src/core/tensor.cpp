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
}