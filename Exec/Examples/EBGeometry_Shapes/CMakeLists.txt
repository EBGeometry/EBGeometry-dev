cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(EBGEOMETRY_EXAMPLE_SHAPES LANGUAGES CXX)

  option(ENABLE_TESTS "" OFF)
  option(ENABLE_EXAMPLES "" OFF)  

  include(${CMAKE_CURRENT_SOURCE_DIR}/../../../CMakeLists.txt)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../..)  
endif()

add_executable(sphere.out sphere.cpp)
if(ENABLE_CUDA)
  set_source_files_properties(sphere.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(sphere.out PROPERTIES CUDA_ARCHITECTURES "native")
  set_target_properties(sphere.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
