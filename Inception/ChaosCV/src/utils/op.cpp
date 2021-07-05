#include "utils/op.hpp"

#include "dnn/layer.hpp"
#include "dnn/layers/binary_op.hpp"
#include "dnn/layers/diag.hpp"
#include "dnn/layers/gemm.hpp"
#include "dnn/layers/invert.hpp"
#include "dnn/layers/permute.hpp"
#include "dnn/layers/reduce.hpp"
#include "dnn/layers/svd.hpp"
#include "dnn/layers/var.hpp"

namespace chaos
{
	inline namespace op
	{
		template<class Op>
		class Operator
		{
		public:
			static Op& Create()
			{
				static Op op;
				return op;
			}

			template<class...Blobs>
			Tensor operator()(const Blobs&...blobs) const
			{
				std::vector<Tensor> bottoms = { {blobs}... };
				std::vector<Tensor> tops(1);
				layer->Forward(bottoms, tops);
				return tops[0];
			}

			Operator& operator[](const std::string& name) { pname = name; return *this; }
			void operator=(const std::any& param) { layer->Set(pname, param); }

		protected:
			Operator() = default;
			Ptr<Layer> layer;
			std::string pname;
		};

		class BinaryOp : public Operator<BinaryOp>
		{
		public:
			BinaryOp() { layer = std::make_shared<dnn::BinaryOp>(); }
		};

		class Diag : public Operator<op::Diag>
		{
		public:
			Diag() { layer = std::make_shared<dnn::Diag>(); }
		};

		class GEMM : public Operator<GEMM>
		{
		public:
			GEMM() { layer = std::make_shared<dnn::GEMM>(); }
		};

		class Invert : public Operator<Invert>
		{
		public:
			Invert() { layer = std::make_shared<dnn::Invert>(); }
		};

		class Permute : public Operator<Permute>
		{
		public:
			Permute() { layer = std::make_shared<dnn::Permute>(); }
		};

		class Reduce : public Operator<Reduce>
		{
		public:
			Reduce() { layer = std::make_shared<dnn::Reduce>(); }
		};

		class SVD : public Operator<SVD>
		{
		public:
			SVD() { layer = std::make_shared<dnn::SVD>(); }

			auto operator()(const Tensor& a) const
			{
				std::vector<Tensor> bottoms = { a };
				std::vector<Tensor> tops(3);
				layer->Forward(bottoms, tops);
				return std::make_tuple(tops[0], tops[1], tops[2]);
			}
		};

		class Var : public Operator<Var>
		{
		public:
			Var() { layer = std::make_shared<dnn::Var>(); }
		};
	}

	Tensor operator+(const Tensor& a, const Tensor& b)
	{
		auto op = op::BinaryOp::Create();
		op["op_type"] = dnn::BinaryOp::ADD;
		return op(a, b);
	}
	Tensor operator+(float a, const Tensor& b)
	{
		auto op = op::BinaryOp::Create();
		op["op_type"] = dnn::BinaryOp::ADD;
		return op(a, b);
	}
	Tensor operator+(const Tensor& a, float b)
	{
		auto op = op::BinaryOp::Create();
		op["op_type"] = dnn::BinaryOp::ADD;
		return op(a, b);
	}

	Tensor operator-(const Tensor& a, const Tensor& b)
	{
		auto op = op::BinaryOp::Create();
		op["op_type"] = dnn::BinaryOp::SUB;
		return op(a, b);
	}
	Tensor operator-(float a, const Tensor& b)
	{
		auto op = op::BinaryOp::Create();
		op["op_type"] = dnn::BinaryOp::SUB;
		return op(a, b);
	}
	Tensor operator-(const Tensor& a, float b)
	{
		auto op = op::BinaryOp::Create();
		op["op_type"] = dnn::BinaryOp::SUB;
		return op(a, b);
	}

	Tensor operator*(const Tensor& a, const Tensor& b)
	{
		auto op = op::GEMM::Create();
		op["alpha"] = 1.f;
		op["beta"] = 0.f;
		op["transA"] = false;
		op["transB"] = false;
		return op(a, b);
	}
	Tensor operator*(float a, const Tensor& b)
	{
		auto op = op::BinaryOp::Create();
		op["op_type"] = dnn::BinaryOp::MUL;
		return op(a, b);
	}
	Tensor operator*(const Tensor& a, float b)
	{
		auto op = op::BinaryOp::Create();
		op["op_type"] = dnn::BinaryOp::MUL;
		return op(a, b);
	}

	Tensor operator/(const Tensor& a, const Tensor& b) // xB = A; x = A * B^{-1}
	{
		auto op = op::Invert::Create();
		return a * op(b);
	}
	Tensor operator/(float a, const Tensor& b)
	{
		auto op = op::BinaryOp::Create();
		op["op_type"] = dnn::BinaryOp::DIV;
		return op(a, b);
	}
	Tensor operator/(const Tensor& a, float b)
	{
		auto op = op::BinaryOp::Create();
		op["op_type"] = dnn::BinaryOp::DIV;
		return op(a, b);
	}

	Tensor diag(const Tensor& a, int d)
	{
		auto op = op::Diag::Create();
		op["diagonal"] = d;
		return op(a);
	}
	Tensor div(const Tensor& a, const Tensor& b)
	{
		auto op = op::BinaryOp::Create();
		op["op_type"] = dnn::BinaryOp::DIV;
		return op(a, b);
	}
	Tensor max(const Tensor& a, bool all, const Array<int>& vecdim)
	{
		auto op = op::Reduce::Create();
		op["op_type"] = dnn::Reduce::MAX;
		op["all"] = all;
		op["vecdim"] = vecdim;
		return op(a);
	}
	Tensor max(const Tensor& a, const Tensor& b)
	{
		auto op = op::BinaryOp::Create();
		op["op_type"] = dnn::BinaryOp::MAX;
		return op(a, b);
	}
	Tensor mean(const Tensor& a, bool all, const Array<int>& vecdim)
	{
		auto op = op::Reduce::Create();
		op["all"] = all;
		op["vecdim"] = vecdim;
		op["op_type"] = dnn::Reduce::AVG;
		return op(a);
	}
	Tensor min(const Tensor& a, bool all, const Array<int>& vecdim)
	{
		auto op = op::Reduce::Create();
		op["op_type"] = dnn::Reduce::MIN;
		op["all"] = all;
		op["vecdim"] = vecdim;
		return op(a);
	}
	Tensor min(const Tensor& a, const Tensor& b)
	{
		auto op = op::BinaryOp::Create();
		op["op_type"] = dnn::BinaryOp::MIN;
		return op(a, b);
	}
	Tensor mul(const Tensor& a, const Tensor& b)
	{
		auto op = op::BinaryOp::Create();
		op["op_type"] = dnn::BinaryOp::MUL;
		return op(a, b);
	}
	Tensor permute(const Tensor& a, const Array<int> orders)
	{
		auto op = op::Permute::Create();
		op["orders"] = orders;
		return op(a);
	}
	Tensor sum(const Tensor& a, bool all, const Array<int>& vecdim)
	{
		auto op = op::Reduce::Create();
		op["op_type"] = dnn::Reduce::SUM;
		op["all"] = all;
		op["vecdim"] = vecdim;
		return op(a);
	}
	std::tuple<Tensor, Tensor, Tensor> svd(const Tensor& a)
	{
		auto op = op::SVD::Create();
		return op(a);
	}
	Tensor transpose(const Tensor& a)
	{
		auto op = op::Permute::Create();
		op["orders"] = Array<int>{1,0};
		return op(a);
	}
	Tensor var(const Tensor& a, bool all, const Array<int>& vecdim, bool unbias)
	{
		auto op = op::Var::Create();
		op["unbias"] = unbias;
		op["all"] = all;
		op["vecdim"] = vecdim;
		return op(a);
	}
}