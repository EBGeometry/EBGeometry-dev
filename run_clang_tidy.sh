#! /bin/bash

rm -rf build
cmake -B build \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DENABLE_DEBUG=ON \
      -DENABLE_TESTS=OFF \
      -DENABLE_EXAMPLES=ON \
      -DENABLE_CUDA=OFF \
      -DENABLE_HIP=OFF

cmake --build build -j4

clang-tidy --quiet -p build Exec/Examples/EBGeometry_Shapes/*.cpp
