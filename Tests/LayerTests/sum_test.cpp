#include "core.hpp"
#include "dnn/layers/sum.hpp"

namespace chaos
{
	TEST_CLASS(SumTest)
	{
	public:
		SumTest() { sum = std::make_shared<Sum>(); }

		TEST_METHOD(SumVector)
		{
			Tensor v = Tensor::randn(Shape(5U));
			Tensor s;
			sum->Forward(v, s);

			float expect = 0.f;
			for (int i = 0; i < 5; i++) expect += v[i];

			Assert::AreEqual(expect, s[0]);
		}

		TEST_METHOD(SumMatrix)
		{
			Tensor m = Tensor::randn(Shape(3, 4));
			Tensor s1;
			{
				sum->Set("dim", -1);
				sum->Forward(m, s1);

				float expect = 0.f;
				for (int i = 0; i < 12; i++) expect += m[i];

				Assert::AreEqual(expect, s1[0]);
			}

			Tensor s2;
			{
				sum->Set("dim", 0);
				sum->Forward(m, s2);

				Assert::AreEqual(3U, s2.shape[0]);
				for (int i = 0; i < 3; i++)
				{
					float expect = 0.f;
					for (int j = 0; j < 4; j++)
					{
						expect += m.At(i,j);
					}
					Assert::AreEqual(expect, s2[i]);
				}
			}

			Tensor s3;
			{
				sum->Set("dim", 1);
				sum->Forward(m, s3);

				Assert::AreEqual(4U, s3.shape[0]);
				for (int i = 0; i < 4; i++)
				{
					float expect = 0.f;
					for (int j = 0; j < 3; j++)
					{
						expect += m.At(j, i);
					}
					Assert::AreEqual(expect, s3[i]);
				}
			}
		}

		TEST_METHOD(SumTensor)
		{
			Tensor t = Tensor::randn(Shape(2,3,4));

			Tensor s1;
			{
				sum->Set("dim", -1);
				sum->Forward(t, s1);

				float expect = 0.f;
				for (int i = 0; i < 24; i++) expect += t[i];

				Assert::AreEqual(expect, s1[0]);
			}

			Tensor s2;
			{
				sum->Set("dim", 0);
				sum->Forward(t, s2);

				Assert::AreEqual(2U, s2.shape[0]);
				for (int i = 0; i < 2; i++)
				{
					float expect = 0.f;
					for (int j = 0; j < 3; j++)
					{
						for (int k = 0; k < 4; k++)
						{
							expect += t.At(i,j,k);
						}
					}
					Assert::AreEqual(expect, s2[i]);
				}
			}

			Tensor s3;
			{
				sum->Set("dim", 1);
				sum->Forward(t, s3);

				Assert::AreEqual(3U, s3.shape[0]);
				for (int i = 0; i < 3; i++)
				{
					float expect = 0.f;
					for (int j = 0; j < 2; j++)
					{
						for (int k = 0; k < 4; k++)
						{
							expect += t.At(j, i, k);
						}
					}
					Assert::AreEqual(expect, s3[i]);
				}
			}

			Tensor s4;
			{
				sum->Set("dim", 2);
				sum->Forward(t, s4);

				Assert::AreEqual(4U, s4.shape[0]);
				for (int i = 0; i < 4; i++)
				{
					float expect = 0.f;
					for (int j = 0; j < 2; j++)
					{
						for (int k = 0; k < 3; k++)
						{
							expect += t.At(j, k, i);
						}
					}
					Assert::AreEqual(expect, s4[i]);
				}
			}
		}

		Ptr<Layer> sum;
	};
}