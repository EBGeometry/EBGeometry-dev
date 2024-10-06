cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(EBGeometry LANGUAGES CUDA CXX)
enable_language(CUDA)
set (CMAKE_CXX_STANDARD 20)

add_compile_definitions(EBGEOMETRY_ENABLE_CUDA)
add_compile_definitions(EBGEOMETRY_ENABLE_DEBUG)
set(EBGEOMETRY_HOME "/home/robertm/Projects/EBGeometry-dev")

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Use Catch2 for testing
find_package(Catch2 3 REQUIRED)

# Source directories
include_directories(${EBGEOMETRY_HOME})

## VecTests
add_executable(VecTests.out VecTests.cpp)
target_link_libraries(VecTests.out PUBLIC Catch2::Catch2WithMain)
set_source_files_properties(VecTests.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(VecTests.out PROPERTIES CUDA_ARCHITECTURES "native")
set_target_properties(VecTests.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

## VertexTests
add_executable(VertexTests.out VertexTests.cpp)
target_link_libraries(VertexTests.out PUBLIC Catch2::Catch2WithMain)
set_source_files_properties(VertexTests.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(VertexTests.out PROPERTIES CUDA_ARCHITECTURES "native")
set_target_properties(VertexTests.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add tests
enable_testing()

add_test(
  NAME VecTests.out
  COMMAND $<TARGET_FILE:VecTests.out>
)

add_test(
  NAME VertexTests.out
  COMMAND $<TARGET_FILE:VertexTests.out>
)
