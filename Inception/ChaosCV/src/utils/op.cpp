#include "utils/op.hpp"

#include "dnn/layers/binary_op.hpp"
#include "dnn/layers/dot.hpp"
#include "dnn/layers/gemm.hpp"
#include "dnn/layers/permute.hpp"
#include "dnn/layers/sum.hpp"

namespace chaos
{
	class Add : public Operator<Add>
	{
	public:
		Add()
		{
			layer = std::make_shared<dnn::BinaryOp>();
			layer->Set("op_type", dnn::BinaryOp::ADD);
		}

		Tensor operator()(const Tensor& a, const Tensor& b) const
		{
			std::vector<Tensor> tops(1);
			layer->Forward({ a,b }, tops);
			return tops[0];
		}

		Add(const Add&) = delete;
		Add& operator=(const Add&) = delete;
	};

	class Div : public Operator<Div>
	{
	public:
		Div()
		{
			layer = std::make_shared<dnn::BinaryOp>();
			layer->Set("op_type", dnn::BinaryOp::DIV);
		}

		Tensor operator()(const Tensor& a, const Tensor& b) const
		{
			std::vector<Tensor> tops(1);
			layer->Forward({ a,b }, tops);
			return tops[0];
		}

		Div(const Div&) = delete;
		Div& operator=(const Div&) = delete;
	};

	class Dot : public Operator<Dot>
	{
	public:
		Dot()
		{
			layer = std::make_shared<dnn::Dot>();
		}

		Tensor operator()(const Tensor& a, const Tensor& b) const
		{
			std::vector<Tensor> tops(1);
			layer->Forward({ a,b }, tops);
			return tops[0];
		}

		Dot(const Dot&) = delete;
		Dot& operator=(const Dot&) = delete;
	};
	class GEMM : public Operator<GEMM>
	{
	public:
		GEMM()
		{
			layer = std::make_shared<dnn::GEMM>();
		}

		Tensor operator()(const Tensor& a, const Tensor& b) const
		{
			std::vector<Tensor> tops(1);
			layer->Forward({ a,b }, tops);
			return tops[0];
		}

		GEMM(const GEMM&) = delete;
		GEMM& operator=(const GEMM&) = delete;
	};
	class Mul : public Operator<Mul>
	{
	public:
		Mul()
		{
			layer = std::make_shared<dnn::BinaryOp>();
			layer->Set("op_type", dnn::BinaryOp::MUL);
		}

		Tensor operator()(const Tensor& a, const Tensor& b) const
		{
			std::vector<Tensor> tops(1);
			layer->Forward({ a,b }, tops);
			return tops[0];
		}

		Mul(const Mul&) = delete;
		Mul& operator=(const Mul&) = delete;
	};

	class Permute : public Operator<Permute>
	{
	public:
		Permute()
		{
			layer = std::make_shared<dnn::Permute>();
		}

		Tensor operator()(const Tensor& a, const std::vector<uint32>& orders) const
		{
			layer->Set("orders", orders);
			std::vector<Tensor> tops(1);
			layer->Forward({ a }, tops);
			return tops[0];
		}

		Tensor operator()(const Tensor&, const Tensor&) const = delete;
		Permute(const Permute&) = delete;
		Permute& operator=(const Permute&) = delete;
	};

	class Sub : public Operator<Sub>
	{
	public:
		Sub()
		{
			layer = std::make_shared<dnn::BinaryOp>();
			layer->Set("op_type", dnn::BinaryOp::SUB);
		}
		Tensor operator()(const Tensor& a, const Tensor& b) const
		{
			std::vector<Tensor> tops(1);
			layer->Forward({ a,b }, tops);
			return tops[0];
		}

		Sub(const Sub&) = delete;
		Sub& operator=(const Sub&) = delete;
	};

	class Sum : public Operator<Sum>
	{
	public:
		Sum()
		{
			layer = std::make_shared<dnn::Sum>();
		}
		Tensor operator()(const Tensor& a) const
		{
			layer->Set("all", true);
			std::vector<Tensor> tops(1);
			layer->Forward({a}, tops);
			return tops[0];
		}
		Tensor operator()(const Tensor& a, const std::vector<uint32>& vecdim) const
		{
			layer->Set("all", false);
			layer->Set("vecdim", vecdim);
			std::vector<Tensor> tops(1);
			layer->Forward({a}, tops);
			return tops[0];
		}
	};
	


	Tensor operator+(const Tensor& a, const Tensor& b)
	{
		auto& op = Add::GetInstance();
		return op(a, b);
	}
	Tensor operator+(float a, const Tensor& b)
	{
		auto& op = Add::GetInstance();
		return op({ a }, b);
	}
	Tensor operator+(const Tensor& a, float b)
	{
		auto& op = Add::GetInstance();
		return op(a, { b });
	}

	Tensor operator-(const Tensor& a, const Tensor& b)
	{
		auto& op = Sub::GetInstance();
		return op(a, b);
	}
	Tensor operator-(float a, const Tensor& b)
	{
		auto& op = Sub::GetInstance();
		return op({ a }, b);
	}
	Tensor operator-(const Tensor& a, float b)
	{
		auto& op = Sub::GetInstance();
		return op(a, { b });
	}

	Tensor operator*(const Tensor& a, const Tensor& b)
	{
		auto& op = GEMM::GetInstance();
		return op(a, b);
	}
	Tensor operator*(float a, const Tensor& b)
	{
		auto& op = Mul::GetInstance();
		return op({ a }, b);
	}
	Tensor operator*(const Tensor& a, float b)
	{
		auto& op = Mul::GetInstance();
		return op(a, {b});
	}

	Tensor operator/(const Tensor& a, const Tensor& b)
	{
		auto& op = Div::GetInstance();
		return op(a, b);
	}
	Tensor operator/(float a, const Tensor& b)
	{
		auto& op = Div::GetInstance();
		return op({ a }, b);
	}
	Tensor operator/(const Tensor& a, float b)
	{
		auto& op = Div::GetInstance();
		return op(a, { b });
	}

	
	Tensor dot(const Tensor& a, const Tensor& b)
	{
		auto& op = Dot::GetInstance();
		return op(a, b);
	}
	Tensor mul(const Tensor& a, const Tensor& b)
	{
		auto& op = Mul::GetInstance();
		return op(a, b);
	}
	Tensor permute(const Tensor& a, const std::vector<uint32>& orders)
	{
		auto& op = Permute::GetInstance();
		return op(a, orders);
	}
	Tensor transpose(const Tensor& a)
	{
		auto& op = Permute::GetInstance();
		std::vector<uint32> orders = {1,0};
		return op(a, orders);
	}
	Tensor sum(const Tensor& a)
	{
		auto& op = Sum::GetInstance();
		return op(a);
	}
	Tensor sum(const Tensor& a, const std::vector<uint32>& vecdim)
	{
		auto& op = Sum::GetInstance();
		return op(a, vecdim);
	}
}