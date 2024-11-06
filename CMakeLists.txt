cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(EBGeometry LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 20)

# Build settings.
set(EBGEOMETRY_HOME "/home/robertm/Projects/EBGeometry-dev")
set(CMAKE_CXX_FLAGS "-O3 -march=native")
set(ENABLE_DEBUG TRUE)
set(ENABLE_CUDA FALSE)
set(ENABLE_HIP FALSE)
set(ENABLE_TESTS TRUE)

if(ENABLE_DEBUG)
  add_compile_definitions(EBGEOMETRY_ENABLE_DEBUG)
endif()

if(ENABLE_CUDA)
  add_compile_definitions(EBGEOMETRY_ENABLE_CUDA)
  
  enable_language(CUDA)
  project(EBGeometry LANGUAGES CUDA CXX)

  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()
endif()

if(ENABLE_HIP)
  add_compile_definitions(EBGEOMETRY_ENABLE_HIP)
endif()

# Well, here we are. This file should be able to build and run the test suite and, if that passes,
# build and run a few example programs.
if(ENABLE_TESTS)
  add_subdirectory(${EBGEOMETRY_HOME}/Exec/Tests)
endif()

