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
option(ENABLE_TESTS "ENABLE_TESTS" OFF)
option(ENABLE_EXAMPLES "ENABLE_EXAMPLES" OFF)
option(ENABLE_CUDA "ENABLE_CUDA" OFF)
option(ENABLE_HIP "ENABLE_HIP" OFF)
option(ENABLE_DOUBLE "ENABLE_DOUBLE" ON)

# Header-only library
add_library(ebgeometry INTERFACE)
target_include_directories(ebgeometry INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Source directories
include_directories(./)

if(ENABLE_DEBUG)
  add_compile_definitions(EBGEOMETRY_ENABLE_DEBUG)
endif()

if(ENABLE_DOUBLE)
  add_compile_definitions(EBGEOMETRY_USE_DOUBLE)
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

# Enable a dummy translation unit so clang-tidy can analyze headers
option(EBGEOMETRY_TIDY_DUMMY "Generate a dummy TU for clang-tidy" ON)

if(EBGEOMETRY_TIDY_DUMMY)
  # Write a tiny TU into the build tree that includes your single public header
  set(_tidy_dummy "${CMAKE_CURRENT_BINARY_DIR}/tidy_dummy.cpp")
  file(WRITE "${_tidy_dummy}" "#include \"${CMAKE_SOURCE_DIR}/EBGeometry.hpp\"\nint main(){return 0;}\n")

  # Compile it so CMake records flags & includes in compile_commands.json
  add_executable(ebg_tidy_dummy "${_tidy_dummy}")
  target_link_libraries(ebg_tidy_dummy PRIVATE ebgeometry)
  target_include_directories(ebg_tidy_dummy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
