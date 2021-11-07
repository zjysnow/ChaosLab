#include "core/tensor.hpp"

#include <random>

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
		DCHECK_EQ(shape, tensor.shape) << "expect shape=" << shape << " but got " << tensor.shape;
		if (steps == tensor.steps)
		{
			memcpy(tensor.data, data, total() * depth * packing);
		}
		else
		{
			size_t dims = shape.size();
			size_t esize = 1 * depth * packing;
			for (size_t i = 0; i < shape.total(); i++)
			{
				size_t dofst = 0; // dst offset
				size_t sofst = 0; // src offset
				size_t idx = i;
				for (int d = 1; d <= dims; d++)
				{
					size_t k = idx % shape[-d];
					dofst += k * tensor.steps[-d];
					sofst += k * steps[-d];
					idx /= shape[-d];
				}
				memcpy((uchar*)tensor.data + dofst * esize, (const uchar*)data + sofst * esize, esize);
			}
		}
	}

	Tensor Tensor::Clone(Allocator* allocator) const
	{
		Tensor tensor = Tensor(shape, depth, packing, allocator);
		CopyTo(tensor);
		return tensor;
	}

	Tensor Tensor::Cut(const Shape& new_shape, int at) const
	{
		size_t dims = shape.size();

		DCHECK_EQ(new_shape.size(), dims);

		// to check if out-of-range
		int range = shape.total() / new_shape.total();
		DCHECK_LT(at, range) << "expect at < " << range << " but got " << at;

		Shape other = shape;
		for (int i = 0; i < dims; i++)
		{
			if (shape[i] == new_shape[i]) other[i] = 1;
		}

		size_t offset = 0;
		for (int d = 1; d <= dims; d++)
		{
			size_t k = at % other[-d];
			offset += k * steps[-d];
			at /= other[-d];
		}

		Shape sub_shape = Squeeze(new_shape);
		// to squeeze the steps
		Steps sub_steps = Array<int>(sub_shape.size());
		for (int i = 0, j = 0; i < dims; i++)
		{
			if (new_shape[i] != 1) sub_steps[j++] = steps[i];
		}
		return Tensor(sub_shape, depth, packing, (uchar*)data + offset * depth * packing, sub_steps);
	}

	
	Tensor Tensor::row(int at) const
	{
		Shape sub = Array<int>(shape.size(), 1);
		sub[-1] = shape[-1];
		return Cut(sub, at);
	}
	Tensor Tensor::col(int at) const
	{
		Shape sub = Array<int>(shape.size(), 1);
		sub[-2] = shape[-2];
		return Cut(sub, at);
	}
	Tensor Tensor::channel(int at) const
	{
		Shape sub = Array<int>(shape.size(), 1);
		sub[-1] = shape[-1];
		sub[-2] = shape[-2];
		return Cut(sub, at);
	}

	static std::default_random_engine engine;
	Tensor Tensor::randn(const Shape& shape, float mu, float sigma, Allocator* allocator)
	{
		std::normal_distribution<float> normal(mu, sigma);
		Tensor r = Tensor(shape, Depth::D4, Packing::CHW, allocator);
		for (size_t i = 0; i < shape.total(); i++)
		{
			r[i] = normal(engine);
		}
		return r;
	}
	Tensor Tensor::randu(const Shape& shape, float min, float max, Allocator* allocator)
	{
		std::uniform_real_distribution<float> uniform(min, max);
		Tensor r = Tensor(shape, Depth::D4, Packing::CHW, allocator);
		for (size_t i = 0; i < shape.total(); i++)
		{
			r[i] = uniform(engine);
		}
		return r;
	}
	Tensor Tensor::zeros(const Shape& shape, Allocator* allocator)
	{
		Tensor z = Tensor(shape, Depth::D4, Packing::CHW, allocator);
		memset(z.data, 0, z.total() * sizeof(float));
		return z;
	}
	Tensor Tensor::eye(int h, int w, Allocator* allocator)
	{
		Tensor e = Tensor(Shape(h, w), Depth::D4, Packing::CHW, allocator);
		memset(e.data, 0, sizeof(float) * h * w);
		for (size_t i = 0; i < std::min(h, w); i++)
		{
			e[i + i * w] = 1;
		}
		return e;
	}
}