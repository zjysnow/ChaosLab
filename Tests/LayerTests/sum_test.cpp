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

			Assert::AreEqual(expect, s[0], eps, L"sum vector error");
		}

		TEST_METHOD(SumMatrix)
		{
			Tensor m = Tensor::randn(Shape(3, 4));
			Tensor s1;
			{
				sum->Set("all", true);
				sum->Forward(m, s1);

				float expect = 0.f;
				for (int i = 0; i < 12; i++) expect += m[i];

				Assert::AreEqual(expect, s1[0], eps, L"sum matrix all error");
			}

			Tensor s2;
			{
				sum->Set("all", false);
				sum->Set("vecdim", std::vector<uint32>{0});
				sum->Forward(m, s2);

				Assert::AreEqual(4U, s2.shape[1]);
				for (int i = 0; i < 4; i++)
				{
					float expect = 0.f;
					for (int j = 0; j < 3; j++)
					{
						expect += m.At(j,i);
					}
					Assert::AreEqual(expect, s2[i], eps, L"sum matrix vecdim={0} error");
				}
			}

			Tensor s3;
			{
				sum->Set("all", false);
				sum->Set("vecdim", std::vector<uint32>{1});
				sum->Forward(m, s3);

				Assert::AreEqual(3U, s3.shape[0]);
				for (int i = 0; i < 3; i++)
				{
					float expect = 0.f;
					for (int j = 0; j < 4; j++)
					{
						expect += m.At(i, j);
					}
					Assert::AreEqual(expect, s3[i], eps, L"sum matrix vecdim={1} error");
				}
			}
		}

		TEST_METHOD(SumTensor)
		{
			Tensor t = Tensor::randn(Shape(2,3,4));

			Tensor s1;
			{
				sum->Set("all", true);
				sum->Forward(t, s1);

				float expect = 0.f;
				for (int i = 0; i < 24; i++) expect += t[i];

				Assert::AreEqual(expect, s1[0], eps, L"sum tensor all error");
			}

			Tensor s2;
			{
				sum->Set("all", false);
				sum->Set("vecdim", std::vector<uint32>{0}); // (2,3,4)->(1,3,4)
				sum->Forward(t, s2);

				Assert::AreEqual(3U, s2.shape[1]);
				Assert::AreEqual(4U, s2.shape[2]);
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 4; j++)
					{
						float expect = 0.f;
						for (int k = 0; k < 2; k++)
						{
							expect += t.At(k,i,j);
						}
						Assert::AreEqual(expect, s2.At(0, i, j), eps, L"sum tensor vecdim={0} error");
					}
					
				}
			}

			Tensor s3;
			{
				sum->Set("all", false);
				sum->Set("vecdim", std::vector<uint32>{0,1}); // (2,3,4)->(1,1,4)
				sum->Forward(t, s3);

				Assert::AreEqual(4U, s3.shape[2]);
				for (int i = 0; i < 4; i++)
				{
					float expect = 0.f;
					for (int j = 0; j < 2; j++)
					{
						for (int k = 0; k < 3; k++)
						{
							expect += t.At(j,k,i);
						}
					}
					Assert::AreEqual(expect, s3[i], eps, L"sum tensor vecdim={0,1} error");
				}
			}
		}

		Ptr<Layer> sum;
	};
}