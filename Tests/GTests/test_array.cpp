#include "testutil.hpp"

TEST(Array, Create)
{
    constexpr size_t size = 13;
    Array<float> arr = Array<float>(size);
    EXPECT_TRUE(arr.data() != nullptr);
    EXPECT_EQ(arr.size(), size);
}

TEST(Array, Resize)
{
    constexpr size_t size = 17;
    Array<float> arr = Array<float>(size);
    EXPECT_TRUE(arr.data() != nullptr);
    EXPECT_EQ(arr.size(), size);

    constexpr size_t new_size = 23;
    arr.Resize(new_size);
    EXPECT_TRUE(arr.data() != nullptr);
    EXPECT_EQ(arr.size(), new_size);
}

TEST(Array, Add)
{
    Array<float> arr1 = { 1,2,3,4,5 };
    Array<float> arr2 = { 6,7,8,9,10 };
    Array<float> expected = {7,9,11,13,15};

    Array<float> arr3 = arr1 + arr2;

    EXPECT_EQ(arr3.size(), expected.size());
    for (int i = 0; i < expected.size(); i++)
    {
        EXPECT_FLOAT_EQ(arr3[i], expected[i]);
    }
}

TEST(Array, Dot)
{
    Array<float> arr1 = { 1,1,1,1,0 };
    Array<float> arr2 = { 1,1,1,1,8 };

    float d1 = dot(arr1, arr2);

    EXPECT_FLOAT_EQ(d1, 4.f);
}

TEST(Array, Cross)
{
    Array<float> a = {1,2,3};
    Array<float> b = {4,2,1};
    Array<float> c = {3,5,6};

    Array<float> r1 = cross(a, b);
    Array<float> r2 = cross(-1.f * b, a);

    for (int i = 0; i < 3; i++)
    {
        EXPECT_FLOAT_EQ(r1[i], r2[i]);
    }

    Array<float> r3 = cross(a, (b + c));
    Array<float> r4 = cross(a, b) + cross(a, c);

    for (int i = 0; i < 3; i++)
    {
        EXPECT_FLOAT_EQ(r3[i], r4[i]);
    }

    Array<float> r5 = cross(a, a);
    for (int i = 0; i < 3; i++)
    {
        EXPECT_FLOAT_EQ(r5[i], 0.f);
    }
}

TEST(Array, Ranges)
{
    Array<float> a = { 3,1,23,5,23,232,2,5,74,3 };
    Array<float> b = { 23,5,23,232,2 };

    Array<float> r1 = a.ranges(-8, -4);
    Array<float> r2 = a.ranges(2, 6);

    for (int i = 0; i < b.size(); i++)
    {
        EXPECT_FLOAT_EQ(r1[i], b[i]);
        EXPECT_FLOAT_EQ(r2[i], b[i]);
    }
}



int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);

    // Run a specific test only
    //testing::GTEST_FLAG(filter) = "OpenCV.read_image";

    // Exclude a specific test
    //testing::GTEST_FLAG(filter) = "-cvtColorTwoPlane.yuv420sp_to_rgb:-cvtColorTwoPlane.rgb_to_yuv420sp"; // The writing test is broken, so skip it

    return RUN_ALL_TESTS();
}
