#cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
#project(EBGeometry LANGUAGES CXX)
#set (CMAKE_CXX_STANDARD 20)

# Build settings.
#set(EBGEOMETRY_HOME "/home/robertm/Projects/EBGeometry-dev")
#set(CMAKE_CXX_FLAGS "-O3 -march=native")
#set(EBGEOMETRY_DEBUG TRUE)
#set(EBGEOMETRY_CUDA FALSE)
#set(EBGEOMETRY_HIP FALSE)

# Pass options to compiler.
if(DEFINED EBGEOMETRY_DEBUG)
  add_compile_definitions(EBGEOMETRY_ENABLE_DEBUG)
endif()

if(EBGEOMETRY_CUDA)
  add_compile_definitions(EBGEOMETRY_ENABLE_CUDA)
  
  enable_language(CUDA)
  project(EBGeometry LANGUAGES CUDA CXX)

  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()
endif()

if(EBGEOMETRY_HIP)
  add_compile_definitions(EBGEOMETRY_ENABLE_HIP)
endif()

# Use Catch2 for testing
find_package(Catch2 3 REQUIRED)

# Source directories
include_directories(${EBGEOMETRY_HOME})

# VecTests
add_executable(VecTests.out VecTests.cpp)
target_link_libraries(VecTests.out PUBLIC Catch2::Catch2WithMain)
if(EBGEOMETRY_CUDA)
  set_source_files_properties(VecTests.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(VecTests.out PROPERTIES CUDA_ARCHITECTURES "native")
  set_target_properties(VecTests.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# VertexTests
add_executable(VertexTests.out VertexTests.cpp)
target_link_libraries(VertexTests.out PUBLIC Catch2::Catch2WithMain)
if(EBGEOMETRY_CUDA)
  set_source_files_properties(VertexTests.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(VertexTests.out PROPERTIES CUDA_ARCHITECTURES "native")
  set_target_properties(VertexTests.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# MeshParserTests
add_executable(MeshParserTests.out MeshParserTests.cpp)
target_link_libraries(MeshParserTests.out PUBLIC Catch2::Catch2WithMain)
if(EBGEOMETRY_CUDA)
  set_source_files_properties(MeshParserTests.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(MeshParserTests.out PROPERTIES CUDA_ARCHITECTURES "native")
  set_target_properties(MeshParserTests.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

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

add_test(
  NAME MeshParserTests.out
  COMMAND $<TARGET_FILE:MeshParserTests.out>
)
