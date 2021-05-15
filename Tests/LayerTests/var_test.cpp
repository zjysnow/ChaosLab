#include "core.hpp"
#include "dnn/layers/var.hpp"

namespace chaos
{
	TEST_CLASS(VarTest)
	{
	public:
		VarTest() { var = std::make_shared<Var>(); }

		TEST_METHOD(VarVector)
		{
			Tensor v = Tensor::randn(Shape(5u));

			Tensor s;
			var->Forward(v, s);

			float mu = 0.f;
			float expect = 0.f;
			for (int i = 0; i < 5; i++)
			{
				mu += v[i];
			}
			mu /= 5;

			for (int i = 0; i < 5; i++)
			{
				expect += (v[i] - mu) * (v[i] - mu);
			}
			expect /= (5 - 1);

			Assert::AreEqual(expect, s[0], eps);
		}

		Ptr<Layer> var;
	};
}