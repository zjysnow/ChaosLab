
find_package(GTest REQUIRED)
macro(chaoscv_add_test name)
  add_executable(test_${name} test_${name}.cpp)
  target_link_libraries(test_${name} PRIVATE ChaosCV GTest::gtest)
  gtest_add_tests(TARGET test_${name})
endmacro()

chaoscv_add_test(array)
