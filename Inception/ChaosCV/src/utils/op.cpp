#include "utils/op.hpp"

#include "dnn/layer.hpp"
#include "dnn/layers/binary_op.hpp"
#include "dnn/layers/gemm.hpp"

namespace chaos
{
	class Operator
	{
	public:
		template<class Type>
		static Operator& Create() 
		{
			static Operator op;
			op.layer = std::make_shared<Type>();
			return op;
		}

		Tensor operator()(const Tensor& a, const Tensor& b) const
		{
			std::vector<Tensor> tops(1);
			layer->Forward({a, b}, tops);
			return tops[0];
		}

		void Set(const std::string& pname, const std::any& param)
		{
			layer->Set(pname, param);
		}
	private:
		Operator() = default;
		Ptr<Layer> layer;
	};


	Tensor operator+(const Tensor& a, const Tensor& b)
	{
		auto op = Operator::Create<BinaryOp>();
		op.Set("op_type", BinaryOp::ADD);
		return op(a, b);
	}
	Tensor operator+(float a, const Tensor& b)
	{
		auto op = Operator::Create<BinaryOp>();
		op.Set("op_type", BinaryOp::ADD);
		return op({ a }, b);
	}
	Tensor operator+(const Tensor& a, float b)
	{
		auto op = Operator::Create<BinaryOp>();
		op.Set("op_type", BinaryOp::ADD);
		return op(a, { b });
	}

	Tensor operator-(const Tensor& a, const Tensor& b)
	{
		auto op = Operator::Create<BinaryOp>();
		op.Set("op_type", BinaryOp::SUB);
		return op(a, b);
	}
	Tensor operator-(float a, const Tensor& b)
	{
		auto op = Operator::Create<BinaryOp>();
		op.Set("op_type", BinaryOp::SUB);
		return op({ a }, b);
	}
	Tensor operator-(const Tensor& a, float b)
	{
		auto op = Operator::Create<BinaryOp>();
		op.Set("op_type", BinaryOp::SUB);
		return op(a, { b });
	}

	Tensor operator*(const Tensor& a, const Tensor& b)
	{
		auto op = Operator::Create<GEMM>();
		op.Set("alpha", 1.f);
		op.Set("beta", 0.f);
		op.Set("transA", false);
		op.Set("transB", false);
		return op(a, b);
	}
	Tensor operator*(float a, const Tensor& b)
	{
		auto op = Operator::Create<BinaryOp>();
		op.Set("op_type", BinaryOp::MUL);
		return op({ a }, b);
	}
	Tensor operator*(const Tensor& a, float b)
	{
		auto op = Operator::Create<BinaryOp>();
		op.Set("op_type", BinaryOp::MUL);
		return op(a, { b });
	}
}