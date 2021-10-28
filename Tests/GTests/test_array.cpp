#include "testutil.hpp"

TEST(Array, Create)
{
    constexpr size_t size = 13;
    Array<float> arr{size};
    EXPECT_TRUE(arr.data() != nullptr);
    EXPECT_EQ(arr.size(), size);
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
