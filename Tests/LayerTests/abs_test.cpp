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
			abs->vkdev = g_gpu.vkdev;
			abs->CreatePipeline(g_gpu.opt);
		}

		~AbsTest()
		{
			abs->DestroyPipeline(g_gpu.opt);
		}
		
		TEST_METHOD(GpuAbs)
		{
			Tensor b, a = Tensor::randn(Shape(4,6));
			std::vector<Tensor> expected(1);

			std::vector<VulkanTensor> bottom_top_blobs(1);
			g_gpu.cmd.RecordUpload(a, bottom_top_blobs[0], g_gpu.opt);
			abs->Forward(bottom_top_blobs, g_gpu.cmd, g_gpu.opt);
			g_gpu.cmd.RecordDownload(bottom_top_blobs[0], b, g_gpu.opt);
			g_gpu.cmd.SubmitAndWait();
			g_gpu.cmd.Reset();

			abs->Forward({ a }, expected, g_gpu.opt);

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