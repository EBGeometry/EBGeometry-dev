cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(EBGEOMETRY_Example)

  option(ENABLE_TESTS "" OFF)
  option(ENABLE_EXAMPLES "" OFF)  
  
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../.. ebgeometry)
endif()

add_executable(sphere.out sphere.cpp)
target_link_libraries(sphere.out PRIVATE ebgeometry)
if(ENABLE_CUDA)
  set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(sphere.out PROPERTIES CUDA_ARCHITECTURES "native")
  set_target_properties(sphere.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
