
include_directories("${CMAKE_SOURCE_DIR}/Inception/ChaosCV/include/")

find_package(Threads REQUIRED)

macro(chaoscv_add_test name)
  #string(TOLOWER ${class} name)
  add_executable(test_${name} test_${name}.cpp)
  target_link_libraries(test_${name} PRIVATE ChaosCV gtest gtest_main Threads::Threads)
  add_test(NAME test_${name} COMMAND ${CMAKE_COMMAND} -DTEST_EXECUTABLE=$<TARGET_FILE:test_${name}> -P ${CMAKE_SOURCE_DIR}/.github/cmake/run_test.cmake)
  set_property(TARGET test_${name} PROPERTY FOLDER "Tests")
  #gtest_add_tests(TARGET test_${name})
endmacro()

chaoscv_add_test(Array)