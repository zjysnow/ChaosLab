#include "core.hpp"

#include "dnn/layers/decomp.hpp"
#include "utils/op.hpp"

namespace chaos
{
	TEST_CLASS(SVDTest)
	{
	public:
		SVDTest() { svd = std::make_shared<SVD>(); }

		TEST_METHOD(SVDCompute)
		{
			Tensor A = Tensor::randn(Shape(4,5));
			std::vector<Tensor> top(3);
			svd->Forward({ A }, top);

			Tensor& W = top[0];
			Tensor& U = top[1];
			Tensor& Vt = top[2];

			Tensor B = U * diag(W) * Vt;

			for (size_t i = 0; i < 20; i++)
			{
				Assert::AreEqual(A[i], B[i], 1e-5f);
			}
		}

		Ptr<Layer> svd;
	};
}