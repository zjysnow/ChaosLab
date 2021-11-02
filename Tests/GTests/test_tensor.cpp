#include "testutil.hpp"
#include <core/tensor.hpp>

TEST(Tensor, Create)
{
    Tensor tensor1 = Tensor(Shape(3,5), Depth::D4, Packing::CHW);
    EXPECT_TRUE(tensor1.data != nullptr);

    Array<uint16_t> arr = { 1,2,3,4,5 };
    Tensor tensor2 = arr;
    EXPECT_NE(arr.data(), tensor2.data);
    uint16_t* data = (uint16_t*)tensor2.data;
    for (int i = 0; i < 5; i++)
    {
        EXPECT_EQ(data[i], arr[i]);
    }
}

TEST(Tensor, CopyTo)
{
    Array<float> arr1 = { 1,2,3,4,5,6,7,8,9 };
    Tensor t1 = arr1;
    Tensor t2 = t1.Clone();
    EXPECT_EQ(t1.shape, t2.shape);
    for (int i = 0; i < 9; i++)
    {
        EXPECT_EQ(t2[i], arr1[i]);
    }

    Array<float> arr2 = { 1,2,3,0,4,5,6,0,7,8,9,0 };
    Tensor t3 = Tensor(Shape(3, 3), Depth::D4, Packing::CHW, arr2.data(), Steps(4, 1));
    Tensor t4 = t3.Clone();
    EXPECT_EQ(t4.shape, t3.shape);
    for (int i = 0; i < 9; i++)
    {
        EXPECT_EQ(t4[i], arr1[i]);
    }
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
