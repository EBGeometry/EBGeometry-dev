cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

#set(CMAKE_CXX_CLANG_TIDY "clang-tidy")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

project(EBGeometry LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 20)

# Build settings.
set(CMAKE_CXX_FLAGS "-O3 -march=native")

option(ENABLE_DEBUG "ENABLE_DEBUG" ON)
option(ENABLE_TESTS "ENABLE_TESTS" ON)
option(ENABLE_EXAMPLES "ENABLE_EXAMPLES" ON)
option(ENABLE_CUDA "ENABLE_CUDA" OFF)
option(ENABLE_HIP "ENABLE_HIP" OFF)

# Header-only library
add_library(ebgeometry INTERFACE)
target_include_directories(ebgeometry INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Source directories
include_directories(./)

if(ENABLE_DEBUG)
  add_compile_definitions(EBGEOMETRY_ENABLE_DEBUG)
endif()

if(ENABLE_CUDA)
  add_compile_definitions(EBGEOMETRY_ENABLE_CUDA)
  
  enable_language(CUDA)
  project(EBGeometry LANGUAGES CUDA CXX)

  set(CMAKE_CUDA_ARCHITECTURES native)

  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()
endif()

if(ENABLE_HIP)
  add_compile_definitions(EBGEOMETRY_ENABLE_HIP)
endif()

# Compile tests (but do not run them). 
if(ENABLE_TESTS)
  add_subdirectory(./Exec/Tests)
endif()

if(ENABLE_EXAMPLES)
  add_subdirectory(./Exec/Examples/EBGeometry_Shapes)
endif()
