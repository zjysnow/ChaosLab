#include "core.hpp"

namespace chaos
{
	TEST_CLASS(BinaryOpTest)
	{
	public:
		BinaryOpTest()
		{
			binary_op = LayerRegistry::CreateLayer("BinaryOp");
			binary_op->vkdev = g_gpu.vkdev;
			binary_op->CreatePipeline(g_gpu.opt);
		}
		~BinaryOpTest()
		{
			binary_op->DestroyPipeline(g_gpu.opt);
		}

		TEST_METHOD(GpuBinaryOp)
		{
			// broadcast same shape
			{
				Tensor a = Tensor::randn(Shape(2, 3, 4));
				Tensor b = Tensor::randn(Shape(2, 3, 4));
				Tensor c;
				std::vector<Tensor> expected(1);

				std::vector<VulkanTensor> bottoms(2);
				std::vector<VulkanTensor> tops(1);

				g_gpu.cmd.RecordUpload(a, bottoms[0], g_gpu.opt);
				g_gpu.cmd.RecordUpload(b, bottoms[1], g_gpu.opt);

				binary_op->Forward(bottoms, tops, g_gpu.cmd, g_gpu.opt);

				g_gpu.cmd.RecordDownload(tops[0], c, g_gpu.opt);
				g_gpu.cmd.SubmitAndWait();
				g_gpu.cmd.Reset();

				binary_op->Forward({ a,b }, expected, g_gpu.opt);

				for (int i = 0; i < 2; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						for (int k = 0; k < 4; k++)
						{
							Assert::AreEqual(expected[0].At(i, j, k), c.At(i, j, k), std::format(L"broadcast same shape at {},{},{}", i, j, k).data());
						}
					}
				}
			}

			// boradcast different shape
			{
				Tensor a = Tensor::randn(Shape(2, 3, 4));
				Tensor b = Tensor::randn(Shape(3, 4));
				Tensor c;
				std::vector<Tensor> expected(1);

				std::vector<VulkanTensor> bottoms(2);
				std::vector<VulkanTensor> tops(1);

				g_gpu.cmd.RecordUpload(a, bottoms[0], g_gpu.opt);
				g_gpu.cmd.RecordUpload(b, bottoms[1], g_gpu.opt);

				binary_op->Forward(bottoms, tops, g_gpu.cmd, g_gpu.opt);

				g_gpu.cmd.RecordDownload(tops[0], c, g_gpu.opt);
				g_gpu.cmd.SubmitAndWait();
				g_gpu.cmd.Reset();

				binary_op->Forward({ a,b }, expected, g_gpu.opt);

				for (int i = 0; i < 2; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						for (int k = 0; k < 4; k++)
						{
							Assert::AreEqual(expected[0].At(i, j, k), c.At(i, j, k), std::format(L"broadcast different shape at {},{},{}", i,j,k).data());
						}
					}
				}
			}
		}

		Ptr<Layer> binary_op;
	};
}