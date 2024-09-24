cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(EBGeometry LANGUAGES CUDA CXX)
enable_language(CUDA)

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

# Build executable
add_executable(VecTests.out VecTests.cpp)
target_link_libraries(VecTests.out PUBLIC Catch2::Catch2WithMain)
set_source_files_properties(VecTests.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(VecTests.out PROPERTIES CUDA_ARCHITECTURES "native")
set_target_properties(VecTests.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

