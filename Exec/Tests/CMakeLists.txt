include (CTest)

# Use Catch2 for testing
find_package(Catch2 3 REQUIRED)



# VecTests
add_executable(VecTests.out VecTests.cpp)
target_link_libraries(VecTests.out PUBLIC Catch2::Catch2WithMain)
if(ENABLE_CUDA)
  set_source_files_properties(VecTests.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(VecTests.out PROPERTIES CUDA_ARCHITECTURES "native")
  set_target_properties(VecTests.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# VertexTests
add_executable(VertexTests.out VertexTests.cpp)
target_link_libraries(VertexTests.out PUBLIC Catch2::Catch2WithMain)
if(ENABLE_CUDA)
  set_source_files_properties(VertexTests.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(VertexTests.out PROPERTIES CUDA_ARCHITECTURES "native")
  set_target_properties(VertexTests.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# MeshParserTests
add_executable(MeshParserTests.out MeshParserTests.cpp)
target_link_libraries(MeshParserTests.out PUBLIC Catch2::Catch2WithMain)
if(ENABLE_CUDA)
  set_source_files_properties(MeshParserTests.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(MeshParserTests.out PROPERTIES CUDA_ARCHITECTURES "native")
  set_target_properties(MeshParserTests.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Add tests
enable_testing()

add_test(
  NAME VecTests.out
  COMMAND $<TARGET_FILE:VecTests.out>
)

add_test(
  NAME VertexTests.out
  COMMAND $<TARGET_FILE:VertexTests.out>
  )

add_test(
  NAME MeshParserTests.out
  COMMAND $<TARGET_FILE:MeshParserTests.out>
)
