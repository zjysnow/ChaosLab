#include "core.hpp"

namespace chaos
{
	TEST_CLASS(PermuteTest)
	{
	public:
		PermuteTest()
		{
			permute = LayerRegistry::CreateLayer("Permute");
			permute->vkdev = g_env.vkdev;
			permute->CreatePipeline(g_env.opt);
		}

		~PermuteTest()
		{
			permute->DestroyPipeline(g_env.opt);
		}

		TEST_METHOD(GpuPermute)
		{
			// transpose
			{
				Array<int> orders = {1,0};
				permute->Set("orders", orders);
				Tensor a = Tensor::randn(Shape(5,11));
				Tensor p;
				
				std::vector<VulkanTensor> bottoms(1);
				std::vector<VulkanTensor> tops(1);

				g_env.cmd.RecordUpload(a, bottoms[0], g_env.opt);
				permute->Forward(bottoms, tops, g_env.cmd, g_env.opt);
				g_env.cmd.RecordDownload(tops[0], p, g_env.opt);
				g_env.cmd.SubmitAndWait();
				g_env.cmd.Reset();

				std::vector<Tensor> expected(1);
				permute->Forward({a}, expected, g_env.opt);

				for (int i = 0; i < 11; i++)
				{
					for (int j = 0; j < 5; j++)
					{
						Assert::AreEqual(expected[0].At(i, j), p.At(i, j), std::format(L"at {}, {}", i, j).data());
					}
				}
			}

			// not need permute
			{
				Array<int> orders = { 0,1 };
				permute->Set("orders", orders);
				Tensor a = Tensor::randn(Shape(3,4));
				Tensor p;

				std::vector<VulkanTensor> bottoms(1);
				std::vector<VulkanTensor> tops(1);

				g_env.cmd.RecordUpload(a, bottoms[0], g_env.opt);
				permute->Forward(bottoms, tops, g_env.cmd, g_env.opt);
				g_env.cmd.RecordDownload(tops[0], p, g_env.opt);
				g_env.cmd.SubmitAndWait();
				g_env.cmd.Reset();

				std::vector<Tensor> expected(1);
				permute->Forward({ a }, expected, g_env.opt);

				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 4; j++)
					{
						Assert::AreEqual(expected[0].At(i, j), p.At(i, j), std::format(L"at {}, {}", i, j).data());
					}
				}
			}
		}

		Ptr<Layer> permute;
	};
}