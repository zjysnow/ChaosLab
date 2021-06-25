#include "core.hpp"

namespace chaos
{

	TEST_CLASS(SumTest)
	{
	public:
		SumTest()
		{
			sum = LayerRegistry::CreateLayer("Sum");
			sum->vkdev = g_env.vkdev;
			sum->CreatePipeline(g_env.opt);
		}
		~SumTest()
		{
			sum->DestroyPipeline(g_env.opt);
		}

		TEST_METHOD(GpuSum)
		{
			// sum to scalar
			sum->Set("all", true);
			{
				Tensor a = Tensor::randn(Shape(4, 2, 5));
				Tensor s;

				std::vector<VulkanTensor> bottoms(1);
				std::vector<VulkanTensor> tops(1);

				g_env.cmd.RecordUpload(a, bottoms[0], g_env.opt);
				sum->Forward(bottoms, tops, g_env.cmd, g_env.opt);
				g_env.cmd.RecordDownload(tops[0], s, g_env.opt);
				g_env.cmd.SubmitAndWait();
				g_env.cmd.Reset();

				std::vector<Tensor> expected(1);
				sum->Forward({ a }, expected, g_env.opt);

				Assert::AreEqual(expected[0][0], s[0]);
			}

			sum->Set("all", false);
			{
				Array<int> vecdim = {1,2};
				sum->Set("vecdim", vecdim);

				Tensor a = Tensor::randn(Shape(4, 2, 5));
				Tensor s; // 4,1,1

				std::vector<VulkanTensor> bottoms(1);
				std::vector<VulkanTensor> tops(1);

				g_env.cmd.RecordUpload(a, bottoms[0], g_env.opt);
				sum->Forward(bottoms, tops, g_env.cmd, g_env.opt);
				g_env.cmd.RecordDownload(tops[0], s, g_env.opt);
				g_env.cmd.SubmitAndWait();
				g_env.cmd.Reset();

				std::vector<Tensor> expected(1);
				sum->Forward({ a }, expected, g_env.opt);

				for (int i = 0; i < 4; i++)
				{
					Assert::AreEqual(expected[0].At(i, 0, 0), s.At(i, 0, 0), std::format(L"{}", i).data());
				}
			}
		}

		Ptr<Layer> sum;
	};
}