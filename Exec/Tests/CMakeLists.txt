# SPDX-FileCopyrightText: 2025 Robert Marskar
#
# SPDX-License-Identifier: LGPL-3.0-or-later

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

set(TEST_SOURCES
  AnalyticDistanceFunctionsTests.cpp
  MeshParserTests.cpp
  TriangleTests.cpp
  VirtualFunctionTests.cpp
  VecTests.cpp
  VertexTests.cpp
)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(EBGEOMETRY_TESTS LANGUAGES CXX)  
  
  option(ENABLE_TESTS "" OFF)
  option(ENABLE_EXAMPLES "" OFF)
  
  include(${CMAKE_CURRENT_SOURCE_DIR}/../../CMakeLists.txt)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include (CTest)

# Use Catch2 for testing
find_package(Catch2 3 REQUIRED)

foreach(CurTestSrc ${TEST_SOURCES})
  get_filename_component(CurTestExec ${CurTestSrc} NAME_WE)
  add_executable(${CurTestExec} ${CurTestSrc})
  target_link_libraries(${CurTestExec} PUBLIC Catch2::Catch2WithMain)
  target_compile_options(${CurTestExec} PRIVATE -fopt-info-vec-missed)    

  if(ENABLE_CUDA)
    set_source_files_properties(${CurTestSrc} PROPERTIES LANGUAGE CUDA)
    set_target_properties(${CurTestExec} PROPERTIES CUDA_ARCHITECTURES "native")
    set_target_properties(${CurTestExec} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  endif()

  add_test(
    NAME ${CurTestExec} 
    COMMAND $<TARGET_FILE:${CurTestExec}>
  )  
endforeach()
