#include "utils/op.hpp"

#include "dnn/layer.hpp"
#include "dnn/layers/binary_op.hpp"
#include "dnn/layers/gemm.hpp"
#include "dnn/layers/invert.hpp"
#include "dnn/layers/permute.hpp"
#include "dnn/layers/svd.hpp"

namespace chaos::inline op
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
			layer->Forward({a,b}, tops);
			return tops[0];
		}

		void operator()(const Tensor& a, const Tensor& b, Tensor& c) const
		{
			std::vector<Tensor> tops{c};
			layer->Forward({a,b}, tops);
		}

		Operator(const Operator&) = delete;
		Operator& operator=(const Operator&) = delete;
	protected:
		Operator() = default;
		virtual ~Operator() { layer->DestroyPipeline(); }
		Ptr<Layer> layer;
	};

	class Add : public Operator<op::Add>
	{
	public:
		Add() { layer = std::make_shared<dnn::BinaryOp>(dnn::BinaryOp::ADD); layer->CreatePipeline(); }
	};

	class Div : public Operator<op::Div>
	{
	public:
		Div() { layer = std::make_shared<dnn::BinaryOp>(dnn::BinaryOp::DIV); layer->CreatePipeline(); }
	};

	class GEMM : public Operator<op::GEMM>
	{
	public:
		GEMM() { layer = std::make_shared<dnn::GEMM>(); layer->CreatePipeline(); }
		Tensor operator()(const Tensor& a, const Tensor& b) const
		{
			layer->Set("transA", false);
			layer->Set("transB", false);
			layer->Set("alpha", 1.f);
			layer->Set("beta", 0.f);

			std::vector<Tensor> tops(1);
			layer->Forward({a, b}, tops);
			return tops[0];
		}
		void operator()(bool transA, bool transB, const Tensor& a, const Tensor& b, float alpha, Tensor& c, float beta) const
		{
			layer->Set("transA", transA);
			layer->Set("transB", transB);
			layer->Set("alpha", alpha);
			layer->Set("beta", beta);

			std::vector<Tensor> tops{ c };
			layer->Forward({ a, b, c }, tops);
		}
		void operator()(const Tensor& a, const Tensor& b, Tensor& c) const = delete;
	};
	class Invert : public Operator<op::Invert>
	{
	public:
		Invert() { layer = std::make_shared<dnn::Invert>(); layer->CreatePipeline(); }
		void operator()(const Tensor& a, Tensor& b, int method) const
		{
			layer->Set("method", (dnn::Invert::Method)method);
			std::vector<Tensor> tops{ b };
			layer->Forward({ a }, tops);
		}

		Tensor operator()(const Tensor& a, const Tensor& b) const = delete;
		void operator()(const Tensor& a, const Tensor& b, Tensor& c) const = delete;
	};
	class Mul : public Operator<op::Mul>
	{
	public:
		Mul() { layer = std::make_shared<dnn::BinaryOp>(dnn::BinaryOp::MUL); layer->CreatePipeline(); }
	};

	class Sub : public Operator<op::Sub>
	{
	public:
		Sub() { layer = std::make_shared<dnn::BinaryOp>(dnn::BinaryOp::SUB); layer->CreatePipeline(); }
	};
	class SVD : public Operator<op::SVD>
	{
	public:
		SVD() { layer = std::make_shared<dnn::SVD>(); layer->CreatePipeline(); }

		void operator()(const Tensor& a, Tensor& w) const
		{
			layer->Set("uv", dnn::SVD::NO_UV);
			std::vector<Tensor> tops{ w };
			layer->Forward({a}, tops);
		}

		void operator()(const Tensor& a, Tensor& w, Tensor& u, Tensor& vt, bool full_uv) const
		{
			layer->Set("uv", full_uv ? dnn::SVD::FULL_UV : dnn::SVD::SIMPLE_UV);
			std::vector<Tensor> tops{ w, u, vt };
			layer->Forward({a}, tops);
		}

		Tensor operator()(const Tensor&, const Tensor&) const = delete;
		void operator()(const Tensor&, const Tensor&, Tensor&) const = delete;
	};
	class Permute : public Operator<op::Permute>
	{
	public:
		Permute() { layer = std::make_shared<dnn::Permute>(); layer->CreatePipeline(); }

		void operator()(const Tensor& a, const Array<uint32>& orders, Tensor& b)
		{
			layer->Set("orders", orders);
			std::vector<Tensor> tops{b};
			layer->Forward({ a }, tops);
		}

		Tensor operator()(const Tensor&, const Tensor&) const = delete;
		void operator()(const Tensor&, const Tensor&, Tensor&) const = delete;
	};
}

namespace chaos
{
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

	Tensor operator*(const Tensor& a, const Tensor& b)
	{
		auto& op = op::GEMM::Get();
		return op(a, b);
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

	Tensor operator/(const Tensor& a, const Tensor& b)
	{
		auto& op = op::Invert::Get();
		Tensor bt;
		op(b, bt, dnn::Invert::DECOMP_SVD);
		return a * bt;
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
		if (c.empty())
		{
			Shape c_shape = a.shape & b.shape;
			c.Create(c_shape, c_shape.steps(), Depth::D4, Packing::CHW, nullptr);
		}
		auto& op = Add::Get();
		op(a, b, c);
	}
	void div(const Tensor& a, const Tensor& b, Tensor& c)
	{
		if (c.empty())
		{
			Shape c_shape = a.shape & b.shape;
			c.Create(c_shape, c_shape.steps(), Depth::D4, Packing::CHW, nullptr);
		}
		auto& op = Div::Get();
		op(a, b, c);
	}
	void gemm(bool transA, bool transB, const Tensor& a, const Tensor& b, float alpha, Tensor& c, float beta)
	{
		auto& op = op::GEMM::Get();
		if (c.empty())
		{
			uint32 m = a.shape[0];
			uint32 n = a.shape[1];
			uint32 p = b.shape[0];
			uint32 k = b.shape[1];

			if (transA) std::swap(m, n);
			if (transB) std::swap(p, k);

			c.Create(Shape(m, k), Steps(k, 1), Depth::D4, Packing::CHW, nullptr);
		}
		op(transA, transB, a, b, alpha, c, beta);
	}
	void invert(const Tensor& a, Tensor& b, int method)
	{
		if (b.empty())
		{
			b.Create(Shape(a.shape[1], a.shape[0]), Steps(a.shape[0], 1), Depth::D4, Packing::CHW, nullptr);
		}
		auto& op = op::Invert::Get();
		op(a, b, method);
	}
	void mul(const Tensor& a, const Tensor& b, Tensor& c)
	{
		if (c.empty())
		{
			Shape c_shape = a.shape & b.shape;
			c.Create(c_shape, c_shape.steps(), Depth::D4, Packing::CHW, nullptr);
		}
		auto& op = Mul::Get();
		op(a, b, c);
	}
	void sub(const Tensor& a, const Tensor& b, Tensor& c)
	{
		if (c.empty())
		{
			Shape c_shape = a.shape & b.shape;
			c.Create(c_shape, c_shape.steps(), Depth::D4, Packing::CHW, nullptr);
		}
		auto& op = Sub::Get();
		op(a, b, c);
	}
	void svd(const Tensor& a, Tensor& w)
	{
		auto& op = op::SVD::Get();
		if (w.empty())
		{
			w.Create(Shape(std::min(a.shape[0], a.shape[1])), Steps(1), Depth::D4, Packing::CHW, nullptr);
		}
		op(a, w);
	}
	void svd(const Tensor& a, Tensor& w, Tensor& u, Tensor& vt, bool full_uv)
	{
		auto& op = op::SVD::Get();
		uint32 m = a.shape[0], n = a.shape[1];
		bool at = false;
		if (m < n) 
		{
			std::swap(m, n); 
			at = true;
		}
		uint32 urows = full_uv ? m : n;
		Shape u_shape;
		Shape vt_shape;
		if (!at)
		{
			u_shape = { m, urows };
			vt_shape = { n, n };
		}
		else
		{
			u_shape = { n, n };
			vt_shape = { urows, m };
		}

		if (w.empty())
		{
			w.Create(Shape(n), Steps(1), Depth::D4, Packing::CHW, nullptr);
		}
		if (u.empty())
		{
			u.Create(u_shape, u_shape.steps(), Depth::D4, Packing::CHW, nullptr);
		}
		if (vt.empty())
		{
			vt.Create(vt_shape, vt_shape.steps(), Depth::D4, Packing::CHW, nullptr);
		}
		op(a, w, u, vt, full_uv);
	}
	
	void transpose(const Tensor& a, Tensor& b)
	{
		auto& op = op::Permute::Get();
		Array<uint32> orders = {1, 0};
		if (b.empty())
		{
			CHECK_EQ(2, a.shape.size());
			Shape b_shape = { a.shape[1], a.shape[0] };
			b.Create(b_shape, b_shape.steps(), Depth::D4, Packing::CHW, nullptr);
		}
		op(a, orders, b);
	}
	void permute(const Tensor& a, const Array<uint32>& orders, Tensor& b)
	{
		auto& op = op::Permute::Get();
		if (b.empty())
		{
			CHECK_EQ(a.shape.size(), orders.size());
			Shape b_shape = a.shape;
			for (size_t i = 0; i < a.shape.size(); i++) b_shape[i] = a.shape[orders[i]];
			b.Create(b_shape, b_shape.steps(), Depth::D4, Packing::CHW, nullptr);
		}
		op(a, orders, b);
	}
}