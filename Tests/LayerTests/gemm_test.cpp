#include "core.hpp"

namespace chaos
{
	TEST_CLASS(GEMMTest)
	{
	public:
		GEMMTest()
		{
			gemm = LayerRegistry::CreateLayer("GEMM");
			gemm->vkdev = g_env.vkdev;
			gemm->CreatePipeline(g_env.opt);
		}

		~GEMMTest()
		{
			gemm->DestroyPipeline(g_env.opt);
		}

		TEST_METHOD(GpuGEMM)
		{
			// c = a * b;
			{
				Tensor a = Tensor::randn(Shape(3,5));
				Tensor b = Tensor::randn(Shape(5,7));
				Tensor c; // Shape(3,7)

				std::vector<VulkanTensor> bottoms(2);
				std::vector<VulkanTensor> tops(1);

				g_env.cmd.RecordUpload(a, bottoms[0], g_env.opt);
				g_env.cmd.RecordUpload(b, bottoms[1], g_env.opt);

				gemm->Forward(bottoms, tops, g_env.cmd, g_env.opt);

				g_env.cmd.RecordDownload(tops[0], c, g_env.opt);

				g_env.cmd.SubmitAndWait();
				g_env.cmd.Reset();

				std::vector<Tensor> expected(1);
				gemm->Forward({ a,b }, expected, g_env.opt);

				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 7; j++)
					{
						Assert::AreEqual(expected[0].At(i,j), c.At(i,j), eps, std::format(L"c=axb at {},{}", i,j).data());
					}
				}
			}

			// d = 0.2 * a^T * b + 0.7 * c
			{
				gemm->Set("alpha", 0.2f);
				gemm->Set("beta", 0.7f);
				gemm->Set("transA", true);

				Tensor a = Tensor::randn(Shape(5, 3));
				Tensor b = Tensor::randn(Shape(5, 7));
				Tensor c = Tensor::randn(Shape(3, 7)); // Shape(3,7)
				Tensor d;

				std::vector<VulkanTensor> bottoms(3);
				std::vector<VulkanTensor> tops(1);

				g_env.cmd.RecordUpload(a, bottoms[0], g_env.opt);
				g_env.cmd.RecordUpload(b, bottoms[1], g_env.opt);
				g_env.cmd.RecordUpload(c, bottoms[2], g_env.opt);

				gemm->Forward(bottoms, tops, g_env.cmd, g_env.opt);

				g_env.cmd.RecordDownload(tops[0], d, g_env.opt);

				g_env.cmd.SubmitAndWait();
				g_env.cmd.Reset();

				std::vector<Tensor> expected(1);
				gemm->Forward({ a,b,c }, expected, g_env.opt);

				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 7; j++)
					{
						Assert::AreEqual(expected[0].At(i, j), d.At(i, j), eps, std::format(L"c=axb at {},{}", i, j).data());
					}
				}
			}
		}

		Ptr<Layer> gemm;
	};
}