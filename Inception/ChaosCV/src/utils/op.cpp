#include "utils/op.hpp"

#include "dnn/layers/binary_op.hpp"
#include "dnn/layers/dot.hpp"
#include "dnn/layers/gemm.hpp"

namespace chaos
{
	Add::Add() 
	{
		layer = std::make_shared<dnn::BinaryOp>();
		layer->Set("op_type", dnn::BinaryOp::ADD);
	}
	Div::Div()
	{
		layer = std::make_shared<dnn::BinaryOp>();
		layer->Set("op_type", dnn::BinaryOp::DIV);
	}
	Dot::Dot()
	{
		layer = std::make_shared<dnn::Dot>();
	}
	GEMM::GEMM()
	{
		layer = std::make_shared<dnn::GEMM>();
	}
	Tensor GEMM::operator()(const Tensor& a, const Tensor& b)
	{
		layer->Set("alpha", 1.f);
		layer->Set("beta", 0.f);
		layer->Set("transA", false);
		layer->Set("transB", false);
		std::vector<Tensor> tops(1);
		layer->Forward({ a,b }, tops);
		return tops[0];
	}
	Tensor GEMM::operator()(int flag, const Tensor& a, const Tensor& b, float alpha, const Tensor& c, float beta) const
	{
		layer->Set("alpha", alpha);
		layer->Set("beta", beta);
		layer->Set("transA", bool(flag & TRANSA));
		layer->Set("transB", bool(flag & TRANSB));
		layer->Set("transC", bool(flag & TRANSC));
		std::vector<Tensor> tops(1);
		layer->Forward({a,b,c}, tops);
		return tops[0];
	}
	Mul::Mul()
	{
		layer = std::make_shared<dnn::BinaryOp>();
		layer->Set("op_type", dnn::BinaryOp::MUL);
	}
	Sub::Sub()
	{
		layer = std::make_shared<dnn::BinaryOp>();
		layer->Set("op_type", dnn::BinaryOp::SUB);
	}
	
	

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

}