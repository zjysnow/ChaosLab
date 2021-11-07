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

    Tensor tensor3 = tensor1;
    EXPECT_EQ(tensor1.data, tensor3.data);

    tensor3 = tensor2;
    EXPECT_EQ(tensor2.data, tensor3.data);
}

TEST(Tensor, CopyTo)
{
    Tensor t1 = Tensor::randu(Shape{ 4,2,6,1 });
    Tensor t2 = t1.Clone();
    EXPECT_EQ(t1.shape, t2.shape);
    for (int i = 0; i < t1.shape.total(); i++)
    {
        EXPECT_FLOAT_EQ(t2[i], t1[i]);
    }

    Array<float> arr2 = { 1,2,3,0,4,5,6,0,7,8,9,0 };
    Tensor t3 = Tensor(Shape(3, 3), Depth::D4, Packing::CHW, arr2.data(), Steps(4, 1));
    Tensor t4 = t3.Clone();
    EXPECT_EQ(t4.shape, t3.shape);
    for (int i = 0; i < 9; i++)
    {
        EXPECT_FLOAT_EQ(t4[i], t3.At(i));
    }
}

TEST(Tensor, Cut)
{
    Tensor rand = Tensor::randn(Shape(3, 4, 5));

    Tensor row1 = rand.row(1).Clone(); //rand.Cut({1,1,5}, 1).Clone();
    EXPECT_EQ(row1.shape, Shape(5));
    for (int i = 0; i < 5; i++)
    {
        EXPECT_FLOAT_EQ(rand.At(0,1,i), row1.At(i));
    }

    Tensor col1 = rand.col(1).Clone(); //rand.Cut({1,4,1}, 1).Clone();
    EXPECT_EQ(col1.shape, Shape(4));
    for (int i = 0; i < 4; i++)
    {
        EXPECT_FLOAT_EQ(rand.At(0,i,1), col1.At(i));
    }

    Tensor ch1 = rand.channel(1).Clone(); //rand.Cut({ 1,4,5 }, 1);
    EXPECT_EQ(ch1.shape, Shape(4, 5));
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            EXPECT_FLOAT_EQ(rand.At(1, i, j), ch1.At(i, j));
        }
    }
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
