#include "dnn/layers/reduce.hpp"

namespace chaos
{
	inline namespace dnn
	{
		struct ReduceSum
		{
			inline void operator()(const float& x, float& y) const { y = y + x; }
		};

		struct ReduceMax
		{
			inline void operator()(const float& x, float& y) const { y = std::max(x, y); }
		};

		struct ReduceMin
		{
			inline void operator()(const float& x, float& y) const { y = std::min(x, y); }
		};

		template<class Op>
		static inline void Operator(const Tensor& a, float alpha, Tensor& r)
		{
			Op op;
			int dims = (int)a.shape.size();
			for (size_t i = 0; i < a.shape.total(); i++)
			{
				size_t r_idx = 0;
				size_t a_idx = 0;
				size_t idx = i;
				for (int d = dims - 1; d >= 0; d--)
				{
					size_t k = idx % a.shape[d];
					r_idx += (k >= r.shape[d] ? 0 : k) * r.steps[d];
					a_idx += k * a.steps[d];
					idx /= a.shape[d];
				}
				op(alpha * a[a_idx], r[r_idx]);
			}
		}

		Reduce::Reduce() : Layer("Reduce") {}

		void Reduce::Forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const Option& opt) const
		{
			CHECK_EQ(1, bottom_blobs.size()) << "layer Sum expect 1 input but got " << bottom_blobs.size();
			CHECK_EQ(1, top_blobs.size()) << "layer Sum expect 1 output but got " << top_blobs.size();

			const Tensor& A = bottom_blobs[0];
			int dims = (int)A.shape.size();
			CHECK_LE(vecdim.size(), dims) << "dims of A should greater-equal than size of vecdims";

			Shape shape = A.shape;
			float N = 1;
			if (all)
			{
				for (int i = 0; i < dims; i++)
				{
					N *= shape[i];
					shape[i] = 1;
				}
			}
			else
			{
				for (const auto& i : vecdim)
				{
					CHECK_LE(i, dims) << "out of range";
					N *= shape[i];
					shape[i] = 1;
				}
			}
			Steps steps = shape.steps();

			Tensor& R = top_blobs[0];
			if (R.empty()) R.Create(shape, steps, Depth::D4, Packing::CHW, opt.blob_allocator);
			CHECK_EQ(shape, R.shape);
			memset(R.data, 0, R.total() * sizeof(float));

			switch (op_type)
			{
			case SUM:
				Operator<ReduceSum>(A, alpha, R);
				break;
			case AVG:
				Operator<ReduceSum>(A, alpha / N, R);
				break;
			case MAX:
				Operator<ReduceMax>(A, alpha, R);
				break;
			case MIN:
				Operator<ReduceMin>(A, alpha, R);
				break;
			default:
				LOG(FATAL);
				break;
			}
		}

		void Reduce::Set(const std::string& pname, const std::any& param)
		{
			if ("op_type" == pname) op_type = std::any_cast<OpType>(param);
			if ("vecdim" == pname) vecdim = std::any_cast<Array<int>>(param);
			if ("all" == pname) all = std::any_cast<bool>(param);
			if ("alpha" == pname) alpha = std::any_cast<float>(param);
		}
	}
}