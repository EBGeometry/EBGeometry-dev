# SPDX-FileCopyrightText: 2025 Robert Marskar
#
# SPDX-License-Identifier: LGPL-3.0-or-later

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

set(EXAMPLE_SOURCES
  Sphere.cpp
)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(EBGEOMETRY_EXAMPLE_SHAPES LANGUAGES CXX)

  option(ENABLE_TESTS "" OFF)
  option(ENABLE_EXAMPLES "" OFF)  

  include(${CMAKE_CURRENT_SOURCE_DIR}/../../../CMakeLists.txt)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../..)  
endif()

foreach(CurExampleSrc ${EXAMPLE_SOURCES})
  get_filename_component(CurExampleBase ${CurExampleSrc} NAME_WE)
  
  set(CurExampleExec "${CurExampleBase}.out")
  
  add_executable(${CurExampleExec} ${CurExampleSrc})

  if(ENABLE_CUDA)
    set_source_files_properties(${CurExampleSrc} PROPERTIES LANGUAGE CUDA)
    set_target_properties(${CurExampleExec} PROPERTIES CUDA_ARCHITECTURES "native")
    set_target_properties(${CurExampleExec} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  endif()

  add_test(
    NAME ${CurExampleExec} 
    COMMAND $<TARGET_FILE:${CurExampleExec}>
  )  
endforeach()
