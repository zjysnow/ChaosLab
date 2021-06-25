#include "core.hpp"

namespace chaos
{
	//TEST_CLASS(Abs), public Test
	// class className : public ::Microsoft::VisualStudio::CppUnitTestFramework::TestClass<className>
	TEST_CLASS(AbsTest)
	{
	public:
		AbsTest()
		{
			abs = LayerRegistry::CreateLayer("Abs");
			abs->vkdev = g_env.vkdev;
			abs->CreatePipeline(g_env.opt);
		}

		~AbsTest()
		{
			abs->DestroyPipeline(g_env.opt);
		}
		
		TEST_METHOD(GpuAbs)
		{
			Tensor b, a = Tensor::randn(Shape(4,6));
			std::vector<Tensor> expected(1);

			std::vector<VulkanTensor> bottom_top_blobs(1);
			g_env.cmd.RecordUpload(a, bottom_top_blobs[0], g_env.opt);
			abs->Forward(bottom_top_blobs, g_env.cmd, g_env.opt);
			g_env.cmd.RecordDownload(bottom_top_blobs[0], b, g_env.opt);
			g_env.cmd.SubmitAndWait();
			g_env.cmd.Reset();

			abs->Forward({ a }, expected, g_env.opt);

			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < 6; j++)
				{
					Assert::AreEqual(expected[0].At(i, j), b.At(i, j), std::format(L"at {},{}", i, j).data());
				}
			}
		}

		Ptr<Layer> abs;
	};


}