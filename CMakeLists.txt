cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(EBGeometry LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 20)

# Build settings.
set(EBGEOMETRY_HOME "/home/robertm/Projects/EBGeometry-dev")
set(CMAKE_CXX_FLAGS "-O3 -march=native")
set(EBGEOMETRY_DEBUG TRUE)
set(EBGEOMETRY_CUDA FALSE)
set(EBGEOMETRY_HIP FALSE)

# Well, here we are. This file should be able to build and run the test suite and, if that passes,
# build and run a few example programs. 
add_subdirectory(Exec/Tests)
