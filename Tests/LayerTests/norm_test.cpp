#include "core.hpp"
#include "dnn/layers/norm.hpp"

namespace chaos
{
	TEST_CLASS(NormTest)
	{
	public:
		NormTest() { norm = std::make_shared<Norm>(); }

		TEST_METHOD(NormVector)
		{
			Tensor v = { 1.f, 2.f, 3.f, 4.f, 5.f };

			Tensor n;

			norm->Set("p", 2.f);
			norm->Forward(v, n);
			Assert::AreEqual(std::sqrt(55.f), n[0]);

			norm->Set("p", 1.f);
			norm->Forward(v, n);
			Assert::AreEqual(15.f, n[0]);

			norm->Set("p", INFINITY);
			norm->Forward(v, n);
			Assert::AreEqual(5.f, n[0]);

			norm->Set("p", -INFINITY);
			norm->Forward(v, n);
			Assert::AreEqual(1.f, n[0]);
		}

		Ptr<Layer> norm;
	};
}