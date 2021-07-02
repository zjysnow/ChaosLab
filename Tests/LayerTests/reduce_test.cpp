#include "core.hpp"
#include "dnn/layers/reduce.hpp"

namespace chaos
{
	TEST_CLASS(ReduceTest)
	{
	public:
		ReduceTest()
		{
			reduce = LayerRegistry::CreateLayer("Reduce");
			reduce->vkdev = g_env.vkdev;
			reduce->CreatePipeline(g_env.opt);
		}
		~ReduceTest()
		{
			reduce->DestroyPipeline(g_env.opt);
		}

		TEST_METHOD(GpuReduceAVG)
		{
			// mean norm
			{
				reduce->Set("op_type", Reduce::AVG);
				reduce->Set("alpha", 1.5f);

				Tensor a = Tensor::randn(Shape(3, 5, 2));
				Tensor b;
				std::vector<VulkanTensor> bottoms(1);
				std::vector<VulkanTensor> tops(1);

				g_env.cmd.RecordUpload(a, bottoms[0], g_env.opt);

				reduce->Forward(bottoms, tops, g_env.cmd, g_env.opt);

				g_env.cmd.RecordDownload(tops[0], b, g_env.opt);

				g_env.cmd.SubmitAndWait();
				g_env.cmd.Reset();

				std::vector<Tensor> expected(1);
				reduce->Forward({ a }, expected, g_env.opt);

				for (int i = 0; i < 5; i++)
				{
					for (int j = 0; j < 2; j++)
					{
						Assert::AreEqual(expected[0].At(0,i,j), b.At(0,i,j), eps, std::format(L"at {},{}", i, j).data());
					}
				}
			}
		}

		Ptr<Layer> reduce;
	};
}