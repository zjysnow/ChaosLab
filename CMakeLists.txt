cmake_minimum_required(VERSION 3.15)

# Disable inplace builds to prevent source tree corruption.
if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "FATAL: Building inplace are not allowed. You should create a separate directory for Building.")
endif()

# Set cmake_install_prefix path
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()

if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Set build type")
endif()

project(ChaosCV)

if( (CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  add_compile_options(-Wextra -Wall -Wno-unused)
endif()
set(CMAKE_CXX_STANDARD 20)

add_definitions(-D CHAOS_EXPORT)

#file(GLOB_RECURSE CHAOSCV_SRC ${CMAKE_SOURCE_DIR}/Inception/ChaosCV/src/*)
#file(GLOB_RECURSE CHAOSCV_INC ${CMAKE_SOURCE_DIR}/Inception/ChaosCV/include/*)

# add core
#add_subdirectory(${CMAKE_SOURCE_DIR}/Inception/ChaosCV/src/core)

aux_source_directory(${CMAKE_SOURCE_DIR}/Inception/ChaosCV/src/core CHAOSCV_CORE_SRC)

# Build NN shared library
add_library(
  ${PROJECT_NAME}
  SHARED
  ${CHAOSCV_CORE_SRC}
)
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC 
  ${CMAKE_SOURCE_DIR}/Inception/ChaosCV/include
)
