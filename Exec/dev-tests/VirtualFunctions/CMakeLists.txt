cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(EBGeometry LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 20)

# Build settings.
set(CMAKE_CXX_FLAGS "-O3 -march=native")
set(ENABLE_DEBUG TRUE)
set(ENABLE_CUDA FALSE)
set(ENABLE_HIP FALSE)
set(ENABLE_TESTS TRUE)

# Source directories
include_directories(./)

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

set(EBGEOMETRY_HOME "/home/robertm/Projects/EBGeometry-dev")

# Source directories
include_directories(${EBGEOMETRY_HOME})

# Build executable
add_executable(main.out main.cpp)

if(ENABLE_CUDA)
  set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(main.out PROPERTIES CUDA_ARCHITECTURES "native")
  set_target_properties(main.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
