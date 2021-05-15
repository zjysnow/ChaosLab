#include "core.hpp"
#include "dnn/layers/mean.hpp"

namespace chaos
{
	TEST_CLASS(MeanTest)
	{
	public:
		MeanTest() { mean = std::make_shared<Mean>(); }

		TEST_METHOD(MeanVector)
		{
			Tensor v = Tensor::randn(Shape(5U));
			Tensor a;
			mean->Forward(v, a);

			float expect = 0.f;
			for (int i = 0; i < 5; i++) expect += v[i];
			expect /= 5;

			Assert::AreEqual(expect, a[0], eps);
		}

		TEST_METHOD(MeanMatrix)
		{
			Tensor m = Tensor::randn(Shape(3,4));

			Tensor a1;
			{
				mean->Set("all", true);
				mean->Forward(m, a1);

				float expect = 0.f;
				for (int i = 0; i < 12; i++) expect += m[i];
				expect /= 12;

				Assert::AreEqual(expect, a1[0], eps, L"mean matrix all");
			}

			Tensor a2;
			{
				mean->Set("all", false);
				mean->Set("vecdim", std::vector<uint32>{1}); // (3,4)->(3,1)
				mean->Forward(m, a2);

				Assert::AreEqual(3U, a2.shape[0]);
				for (int i = 0; i < 3; i++)
				{
					float expect = 0.f;
					for (int j = 0; j < 4; j++)
					{
						expect += m.At(i,j);
					}
					expect /= 4;
					Assert::AreEqual(expect, a2.At(i,0), eps, L"mean matrix vecdim={1}");
				}
			}
		}

		TEST_METHOD(MeanTensor)
		{
			Tensor t = Tensor::randn(Shape(3,4,5));

			Tensor a1;
			{
				mean->Set("all", true);
				mean->Forward(t, a1);

				float expect = 0.f;
				for (int i = 0; i < 60; i++) expect += t[i];
				expect /= 60;

				Assert::AreEqual(expect, a1[0], eps);
			}

			Tensor a2;
			{
				mean->Set("all", false);
				mean->Set("vecdim", std::vector<uint32>{2}); // (3,4,5)->(3,4,1)
				mean->Forward(t, a2);

				Assert::AreEqual(3U, a2.shape[0]);
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 4; j++)
					{
						float expect = 0.f;
						for (int k = 0; k < 5; k++)
						{
							expect += t.At(i, j, k);
						}
						expect /= 5;
						Assert::AreEqual(expect, a2.At(i,j,0), eps);
					}
					
				}
			}

			Tensor a3;
			{
				mean->Set("all", false);
				mean->Set("vecdim", std::vector<uint32>{1,2}); // (3,4,5)->(3,1,1)
				mean->Forward(t, a3);

				Assert::AreEqual(3U, a3.shape[0]);
				for (int i = 0; i < 3; i++)
				{
					float expect = 0.f;
					for (int j = 0; j < 4; j++)
					{
						for (int k = 0; k < 5; k++)
						{
							expect += t.At(i, j, k);
						}
					}
					expect /= 20;
					Assert::AreEqual(expect, a3[i], eps);
				}
			}
		}

		Ptr<Layer> mean;
	};

}