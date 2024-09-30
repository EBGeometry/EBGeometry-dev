cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(EBGeometry LANGUAGES CUDA CXX)
project(EBGeometry LANGUAGES CXX)
enable_language(CUDA)
set (CMAKE_CXX_STANDARD 20)

add_compile_definitions(EBGEOMETRY_ENABLE_CUDA)
add_compile_definitions(EBGEOMETRY_ENABLE_DEBUG)

set(EBGEOMETRY_HOME "/home/robertm/Projects/EBGeometry-dev")

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Source directories
include_directories(${EBGEOMETRY_HOME})

# Build executable
add_executable(main.out main.cpp)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(main.out PROPERTIES CUDA_ARCHITECTURES "native")
set_target_properties(main.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
