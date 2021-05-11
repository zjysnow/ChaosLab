#include "utils/op.hpp"

#include "dnn/layers/binary_op.hpp"
#include "dnn/layers/cross.hpp"
#include "dnn/layers/diag.hpp"
#include "dnn/layers/dot.hpp"
#include "dnn/layers/gemm.hpp"
#include "dnn/layers/mean.hpp"
#include "dnn/layers/norm.hpp"
#include "dnn/layers/normalize.hpp"
#include "dnn/layers/permute.hpp"

namespace chaos
{
	Ptr<Layer> Operator::binary_op = std::make_shared<chaos::BinaryOp>();
	Ptr<Layer> Operator::cross = std::make_shared<chaos::Cross>();
	Ptr<Layer> Operator::dot = std::make_shared<chaos::Dot>();
	Ptr<Layer> Operator::diag = std::make_shared<chaos::Diag>();
	Ptr<Layer> Operator::gemm = std::make_shared<chaos::GEMM>();
	Ptr<Layer> Operator::mean = std::make_shared<chaos::Mean>();
	Ptr<Layer> Operator::norm = std::make_shared<chaos::Norm>();
	Ptr<Layer> Operator::normalize = std::make_shared<chaos::Normalize>();
	Ptr<Layer> Operator::permute = std::make_shared<chaos::Permute>();

	inline void Operator::Add(const Tensor& a, const Tensor& b, Tensor& c)
	{
		binary_op->Set("op_type", BinaryOp::ADD);

		Shape shape = a.shape.total() > b.shape.total() ? a.shape : b.shape;
		if (c.empty()) c.Create(shape, shape.steps(), DataType::D4, Packing::CHW, nullptr);

		std::vector<Tensor> tops{c};
		binary_op->Forward({ a,b }, tops);
	}

	void Operator::Add(float a, const Tensor& b, Tensor& c)
	{
		binary_op->Set("op_type", BinaryOp::ADD);

		if (c.empty()) c.Create(b.shape, b.shape.steps(), DataType::D4, Packing::CHW, nullptr);

		Tensor a_ = { a };
		std::vector<Tensor> tops{ c };
		binary_op->Forward({ a_,b }, tops);
	}

	void Operator::Cross(const Tensor& a, const Tensor& b, Tensor& c)
	{
		if (c.empty()) c.Create(Shape(3u), { 1 }, DataType::D4, Packing::CHW, nullptr);
		std::vector<Tensor> top{ c };
		cross->Forward({a,b}, top);
	}

	void Operator::Diag(const Tensor& a, Tensor& b)
	{
		diag->Forward(a, b);
	}
	void Operator::Div(const Tensor& a, const Tensor& b, Tensor& c)
	{
		binary_op->Set("op_type", BinaryOp::DIV);

		Shape shape = a.shape.total() > b.shape.total() ? a.shape : b.shape;
		if (c.empty()) c.Create(shape, shape.steps(), DataType::D4, Packing::CHW, nullptr);

		std::vector<Tensor> tops{ c };
		binary_op->Forward({ a,b }, tops);
	}
	void Operator::Div(float a, const Tensor& b, Tensor& c)
	{
		binary_op->Set("op_type", BinaryOp::DIV);

		if (c.empty()) c.Create(b.shape, b.shape.steps(), DataType::D4, Packing::CHW, nullptr);

		Tensor a_ = { a };
		std::vector<Tensor> tops{ c };
		binary_op->Forward({ a_,b }, tops);
	}
	void Operator::Dot(const Tensor& a, const Tensor& b, Tensor& c)
	{
		if (c.empty()) c.Create({ 1 }, {1}, DataType::D4, Packing::CHW, nullptr);
		std::vector<Tensor> top{ c };
		dot->Forward({a,b}, top);
	}

	void Operator::GEMM(const Tensor& a, const Tensor& b, float alpha, Tensor& c, float beta)
	{
		gemm->Set("alpha", alpha);
		gemm->Set("beta", beta);

		CHECK_EQ(2, a.shape.dims) << "input A must be a matrix";
		CHECK_EQ(2, b.shape.dims) << "input B must be a matrix";
		Shape shape = { a.shape[0], b.shape[1] };

		if (c.empty()) c.Create(shape, shape.steps(), DataType::D4, Packing::CHW, nullptr);
		std::vector<Tensor> top{ c };
		gemm->Forward({a,b,c}, top);
	}

	void Operator::Mean(const Tensor& a, int dim, Tensor& m)
	{
		mean->Set("dim", dim);
		mean->Forward(a, m);
	}

	void Operator::Mul(const Tensor& a, const Tensor& b, Tensor& c)
	{
		binary_op->Set("op_type", BinaryOp::MUL);

		Shape shape = a.shape.total() > b.shape.total() ? a.shape : b.shape;
		if (c.empty()) c.Create(shape, shape.steps(), DataType::D4, Packing::CHW, nullptr);

		std::vector<Tensor> tops{ c };
		binary_op->Forward({ a,b }, tops);
	}
	void Operator::Mul(float a, const Tensor& b, Tensor& c)
	{
		binary_op->Set("op_type", BinaryOp::MUL);

		if (c.empty()) c.Create(b.shape, b.shape.steps(), DataType::D4, Packing::CHW, nullptr);

		Tensor a_ = { a };
		std::vector<Tensor> tops{ c };
		binary_op->Forward({ a_,b }, tops);
	}

	void Operator::Norm(const Tensor& a, float p, Tensor& n)
	{
		norm->Set("p", p);
		norm->Forward(a, n);
	}
	void Operator::Normalize(const Tensor& a, int dim, const std::string& method, float p1, float p2, Tensor& n)
	{
		if ("norm" == method)
		{
			normalize->Set("method", Normalize::NORM);
			normalize->Set("p1", p1);
		}
		else if ("zscore" == method)
		{
			normalize->Set("method", Normalize::ZSCORE);
		}
		else if ("range" == method)
		{
			normalize->Set("method", Normalize::RANGE);
			normalize->Set("p1", p1);
			normalize->Set("p2", p2);
		}
		else
		{
			LOG(FATAL) << "method should be 'norm', 'zscore' or 'range'";
		}
		normalize->Set("dim", dim);

		normalize->Forward(a, n);
	}

	void Operator::Sub(const Tensor& a, const Tensor& b, Tensor& c)
	{
		binary_op->Set("op_type", BinaryOp::SUB);

		Shape shape = a.shape.total() > b.shape.total() ? a.shape : b.shape;
		if (c.empty()) c.Create(shape, shape.steps(), DataType::D4, Packing::CHW, nullptr);

		std::vector<Tensor> tops{ c };
		binary_op->Forward({ a,b }, tops);
	}
	void Operator::Sub(float a, const Tensor& b, Tensor& c)
	{
		binary_op->Set("op_type", BinaryOp::SUB);

		if (c.empty()) c.Create(b.shape, b.shape.steps(), DataType::D4, Packing::CHW, nullptr);

		Tensor a_ = { a };
		std::vector<Tensor> tops{ c };
		binary_op->Forward({ a_,b }, tops);
	}

	void Operator::Transpose(const Tensor& a, Tensor& b)
	{
		CHECK_EQ(2, a.shape.dims) << "a must be a matrix";
		std::vector<uint32> orders = { 1,0 };
		permute->Set("orders", orders);
		permute->Forward(a, b);
	}




	inline Tensor operator+(const Tensor& lhs, const Tensor& rhs)
	{
		Tensor ret;
		Operator::Add(lhs, rhs, ret);
		return ret;
	}
	inline Tensor operator+(const Tensor& lhs, float rhs)
	{
		Tensor ret;
		Operator::Add(rhs, lhs, ret);
		return ret;
	}
	inline Tensor operator+(float lhs, const Tensor& rhs)
	{
		Tensor ret;
		Operator::Add(lhs, rhs, ret);
		return ret;
	}

	inline Tensor operator-(const Tensor& lhs, const Tensor& rhs)
	{
		Tensor ret;
		Operator::Sub(lhs, rhs, ret);
		return ret;
	}
	inline Tensor operator-(const Tensor& lhs, float rhs)
	{
		Tensor ret;
		Operator::Add(-rhs, lhs, ret);
		return ret;
	}
	inline Tensor operator-(float lhs, const Tensor& rhs)
	{
		Tensor ret;
		Operator::Sub(lhs, rhs, ret);
		return ret;

	}

	inline Tensor operator*(const Tensor& lhs, const Tensor& rhs)
	{
		Tensor ret;
		Operator::GEMM(lhs, rhs, 1.f, ret, 0.f);
		return ret;
	}
	inline Tensor operator*(const Tensor& lhs, float rhs)
	{
		Tensor ret;
		Operator::Mul(rhs, lhs, ret);
		return ret;
	}
	inline Tensor operator*(float lhs, const Tensor& rhs)
	{
		Tensor ret;
		Operator::Mul(lhs, rhs, ret);
		return ret;
	}

	inline Tensor operator/(const Tensor& lhs, const Tensor& rhs)
	{
		Tensor ret;
		Operator::Div(lhs, rhs, ret);
		return ret;
	}
	inline Tensor operator/(const Tensor& lhs, float rhs)
	{
		Tensor ret;
		Operator::Mul(1.f/rhs, lhs, ret);
		return ret;
	}
	inline Tensor operator/(float lhs, const Tensor& rhs)
	{
		Tensor ret;
		Operator::Div(lhs, rhs, ret);
		return ret;
	}

	inline Tensor cross(const Tensor& a, const Tensor& b)
	{
		Tensor ret;
		Operator::Cross(a, b, ret);
		return ret;
	}
	Tensor diag(const Tensor& a, int k)
	{
		Tensor ret;
		Operator::Diag(a, ret);
		return ret;
	}
	inline float dot(const Tensor& a, const Tensor& b)
	{
		Tensor ret;
		Operator::Dot(a, b, ret);
		return ret[0];
	}
	inline float mean(const Tensor& a)
	{
		Tensor ret;
		Operator::Mean(a, -1, ret);
		return ret[0];
	}
	inline float norm(const Tensor& a, float p)
	{
		Tensor ret;
		Operator::Norm(a, p, ret);
		return ret[0];
	}
	inline Tensor normalize(const Tensor& a, int dim, const std::string& method, float p1, float p2)
	{
		Tensor ret;
		Operator::Normalize(a, dim, method, p1, p2, ret);
		return ret;
	}
}