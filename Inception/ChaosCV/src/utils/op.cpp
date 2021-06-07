#include "utils/op.hpp"

#include "dnn/layer.hpp"
#include "dnn/layers/binary_op.hpp"

namespace chaos
{
	template<class Op>
	class Operator
	{
	public:
		static Op& Get()
		{
			static Op op;
			return op;
		}

		Tensor operator()(const Tensor& a, const Tensor& b) const
		{
			std::vector<Tensor> tops(1);
			layer->Forward({a, b}, tops);
			return tops[0];
		}
		void operator()(const Tensor& a, const Tensor& b, Tensor& c) const
		{
			std::vector<Tensor> tops = { c };
			layer->Forward({ a,b }, tops);
		}

		Operator(const Operator&) = delete;
		Operator& operator=(const Operator&) = delete;
	protected:
		Operator() = default;
		virtual ~Operator() = default;
		Ptr<Layer> layer;
	};

	class Add : public Operator<Add>
	{
	public:
		Add() { layer = std::make_shared<BinaryOp>(BinaryOp::ADD); }
	};

	class Div : public Operator<Div>
	{
	public:
		Div() { layer = std::make_shared<BinaryOp>(BinaryOp::DIV); }
	};

	class Mul : public Operator<Mul>
	{
	public:
		Mul() { layer = std::make_shared<BinaryOp>(BinaryOp::MUL); }
	};

	class Sub : public Operator<Sub>
	{
	public:
		Sub() { layer = std::make_shared<BinaryOp>(BinaryOp::SUB); }
	};

	Tensor operator+(const Tensor& a, const Tensor& b)
	{
		auto& op = Add::Get();
		return op(a, b);
	}
	Tensor operator+(float a, const Tensor& b)
	{
		auto& op = Add::Get();
		return op({ a }, b);
	}
	Tensor operator+(const Tensor& a, float b)
	{
		auto& op = Add::Get();
		return op(a, { b });
	}

	Tensor operator-(const Tensor& a, const Tensor& b)
	{
		auto& op = Sub::Get();
		return op(a, b);
	}
	Tensor operator-(float a, const Tensor& b)
	{
		auto& op = Sub::Get();
		return op({ a }, b);
	}
	Tensor operator-(const Tensor& a, float b)
	{
		auto& op = Sub::Get();
		return op(a, { b });
	}

	Tensor operator*(float a, const Tensor& b)
	{
		auto& op = Mul::Get();
		return op({ a }, b);
	}
	Tensor operator*(const Tensor& a, float b)
	{
		auto& op = Mul::Get();
		return op(a, { b });
	}

	Tensor operator/(float a, const Tensor& b)
	{
		auto& op = Div::Get();
		return op({ a }, b);
	}
	Tensor operator/(const Tensor& a, float b)
	{
		auto& op = Div::Get();
		return op(a, { b });
	}


	void add(const Tensor& a, const Tensor& b, Tensor& c)
	{
		auto& op = Add::Get();
		op(a, b, c);
	}
	void div(const Tensor& a, const Tensor& b, Tensor& c)
	{
		auto& op = Div::Get();
		op(a, b, c);
	}
	void mul(const Tensor& a, const Tensor& b, Tensor& c)
	{
		auto& op = Mul::Get();
		op(a, b, c);
	}
	void sub(const Tensor& a, const Tensor& b, Tensor& c)
	{
		auto& op = Sub::Get();
		op(a, b, c);
	}
}