cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(EBGeometry LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 20)

# Build settings.
set(EBGEOMETRY_HOME "/home/robertm/Projects/EBGeometry-dev")
set(CMAKE_CXX_FLAGS "-O3 -march=native")
set(ENABLE_DEBUG TRUE)
set(ENABLE_CUDA FALSE)
set(ENABLE_HIP FALSE)

# Build and run test files
set(ENABLE_TESTS FALSE)

# Well, here we are. This file should be able to build and run the test suite and, if that passes,
# build and run a few example programs.
if(ENABLE_TESTS)
#  add_subdirectory(Exec/Tests)
endif()
