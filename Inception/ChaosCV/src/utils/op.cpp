#include "utils/op.hpp"

#include "dnn/layers/binary_op.hpp"
#include "dnn/layers/cross.hpp"
#include "dnn/layers/dot.hpp"
#include "dnn/layers/normalize.hpp"

namespace chaos
{
	Ptr<Layer> Operator::binary_op = std::make_shared<BinaryOp>();
	Ptr<Layer> Operator::cross = std::make_shared<chaos::Cross>();
	Ptr<Layer> Operator::dot = std::make_shared<chaos::Dot>();
	Ptr<Layer> Operator::normalize = std::make_shared<chaos::Normalize>();

	inline Tensor Operator::Add(const Tensor& a, const Tensor& b)
	{
		binary_op->Set("op_type", BinaryOp::ADD);
		std::vector<Tensor> c(1);
		binary_op->Forward({a,b}, c);
		return c[0];
	}

	inline Tensor Operator::Sub(const Tensor& a, const Tensor& b)
	{
		binary_op->Set("op_type", BinaryOp::SUB);
		std::vector<Tensor> c(1);
		binary_op->Forward({a,b}, c);
		return c[0];
	}

	inline Tensor Operator::Mul(const Tensor& a, const Tensor& b)
	{
		binary_op->Set("op_type", BinaryOp::MUL);
		std::vector<Tensor> c(1);
		binary_op->Forward({ a,b }, c);
		return c[0];
	}
	inline Tensor Operator::Div(const Tensor& a, const Tensor& b)
	{
		binary_op->Set("op_type", BinaryOp::DIV);
		std::vector<Tensor> c(1);
		binary_op->Forward({ a,b }, c);
		return c[0];
	}

	inline float Operator::Dot(const Tensor& a, const Tensor& b)
	{
		std::vector<Tensor> c(1);
		dot->Forward({ a,b }, c);
		return c[0][0];
	}

	Tensor Operator::Mul(const Tensor& a, float b)
	{
		binary_op->Set("op_type", BinaryOp::MUL);
		Tensor b_ = Tensor(Shape(1u), DataType::D4, Packing::CHW, &b);
		std::vector<Tensor> c(1);
		binary_op->Forward({ a,b_ }, c);
		return c[0];
	}

	inline Tensor Operator::Cross(const Tensor& a, const Tensor& b)
	{
		std::vector<Tensor> c(1);
		cross->Forward({a,b}, c);
		return c[0];
	}

	inline Tensor Operator::L2Norm(const Tensor& a)
	{
		normalize->Set("norm_type", Normalize::L2);
		Tensor n;
		normalize->Forward(a, n);
		return n;
	}
}