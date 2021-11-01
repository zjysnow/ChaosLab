
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/")

aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/src/core" CHAOSCV_CORE)

add_library(ChaosCV SHARED ${CHAOSCV_CORE})
#set_target_properties(ChaosCV PROPERTIES DEBUG_POSTFIX "d")

if(CHAOS_COVERAGE)
  target_compile_options(ChaosCV PUBLIC -coverage -fprofile-arcs -ftest-coverage)
  target_link_libraries(ChaosCV PUBLIC -coverage -lgcov)
endif()