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
			// data form Julia
			Tensor v = { -0.602635f,  0.412389f, -0.0208629f, -0.134159f,  0.024829f };
			Tensor n;

			norm->Set("p", 2.f);
			norm->Forward(v, n);
			Assert::AreEqual(0.743158f, n[0], 1E-6f);

			norm->Set("p", 1.f);
			norm->Forward(v, n);
			Assert::AreEqual(1.194874f, n[0], 1E-6f);

			norm->Set("p", 1.2f);
			norm->Forward(v, n);
			Assert::AreEqual(1.001059f, n[0], 1E-6f);

			norm->Set("p", 0.6f);
			norm->Forward(v, n);
			Assert::AreEqual(2.743633f, n[0], 1E-6f);

			norm->Set("p", INFINITY);
			norm->Forward(v, n);
			Assert::AreEqual(0.602635f, n[0], 1E-6f);

			norm->Set("p", -INFINITY);
			norm->Forward(v, n);
			Assert::AreEqual(0.020863f, n[0], 1E-6f);
		}



		Ptr<Layer> norm;
	};
}