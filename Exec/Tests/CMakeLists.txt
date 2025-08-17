cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(EBGEOMETRY_TESTS LANGUAGES CXX)  
  
  option(ENABLE_TESTS "" OFF)
  option(ENABLE_EXAMPLES "" OFF)
  
  include(${CMAKE_CURRENT_SOURCE_DIR}/../../CMakeLists.txt)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

# if(ENABLE_CUDA)
#   enable_language(CUDA)
# endif()

include (CTest)

# Use Catch2 for testing
find_package(Catch2 3 REQUIRED)

# AnalyticDistanceFunctions test
add_executable(AnalyticDistanceFunctionsTests.out AnalyticDistanceFunctionsTests.cpp)
target_link_libraries(AnalyticDistanceFunctionsTests.out PUBLIC Catch2::Catch2WithMain)
if(ENABLE_CUDA)
  set_source_files_properties(AnalyticDistanceFunctionsTests.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(AnalyticDistanceFunctionsTests.out PROPERTIES CUDA_ARCHITECTURES "native")
  set_target_properties(AnalyticDistanceFunctionsTests.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Virtual functions on CPU and GPU
add_executable(VirtualFunctionTests.out VirtualFunctionTests.cpp)
target_link_libraries(VirtualFunctionTests.out PUBLIC Catch2::Catch2WithMain)
if(ENABLE_CUDA)
  set_source_files_properties(VirtualFunctionTests.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(VirtualFunctionTests.out PROPERTIES CUDA_ARCHITECTURES "native")
  set_target_properties(VirtualFunctionTests.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# VecTests
add_executable(VecTests.out VecTests.cpp)
target_link_libraries(VecTests.out PUBLIC Catch2::Catch2WithMain)
if(ENABLE_CUDA)
  set_source_files_properties(VecTests.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(VecTests.out PROPERTIES CUDA_ARCHITECTURES "native")
  set_target_properties(VecTests.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# VertexTests
add_executable(VertexTests.out VertexTests.cpp)
target_link_libraries(VertexTests.out PUBLIC Catch2::Catch2WithMain)
if(ENABLE_CUDA)
  set_source_files_properties(VertexTests.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(VertexTests.out PROPERTIES CUDA_ARCHITECTURES "native")
  set_target_properties(VertexTests.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# TriangleTests
add_executable(TriangleTests.out TriangleTests.cpp)
target_link_libraries(TriangleTests.out PUBLIC Catch2::Catch2WithMain)
if(ENABLE_CUDA)
  set_source_files_properties(TriangleTests.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(TriangleTests.out PROPERTIES CUDA_ARCHITECTURES "native")
  set_target_properties(TriangleTests.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# MeshParserTests
add_executable(MeshParserTests.out MeshParserTests.cpp)
target_link_libraries(MeshParserTests.out PUBLIC Catch2::Catch2WithMain)
if(ENABLE_CUDA)
  set_source_files_properties(MeshParserTests.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(MeshParserTests.out PROPERTIES CUDA_ARCHITECTURES "native")
  set_target_properties(MeshParserTests.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Add tests
enable_testing()

add_test(
  NAME AnalyticDistanceFunctionsTests.out
  COMMAND $<TARGET_FILE:AnalyticDistanceFunctionsTests.out>
)

add_test(
  NAME VirtualFunctionTests.out
  COMMAND $<TARGET_FILE:VirtualFunctionTests.out>
)

add_test(
  NAME VecTests.out
  COMMAND $<TARGET_FILE:VecTests.out>
)

add_test(
  NAME VertexTests.out
  COMMAND $<TARGET_FILE:VertexTests.out>
)

add_test(
  NAME TriangleTests.out
  COMMAND $<TARGET_FILE:TriangleTests.out>
  )

add_test(
  NAME MeshParserTests.out
  COMMAND $<TARGET_FILE:MeshParserTests.out>
)
