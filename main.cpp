// Std includes
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Our includes
#include "EBGeometry.hpp"

using namespace EBGeometry;

EBGEOMETRY_GPU_GLOBAL
void
addNumbers(Vec3* c, const Vec3* const a, const Vec3* const b)
{
  *c = *a + *b;

  return;
}

int
main()
{
  Vec3 v1 = Vec3::one();
  Vec3 v2 = Vec3::one();
  Vec3 v3 = Vec3::one();

  Vec3* d_v1;
  Vec3* d_v2;
  Vec3* d_v3;

  cudaMalloc((void**)&d_v1, sizeof(Vec3));
  cudaMalloc((void**)&d_v2, sizeof(Vec3));
  cudaMalloc((void**)&d_v3, sizeof(Vec3));

  cudaMemcpy(d_v1, &v1, sizeof(Vec3), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v2, &v2, sizeof(Vec3), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v3, &v3, sizeof(Vec3), cudaMemcpyHostToDevice);

  addNumbers<<<1, 1>>>(d_v3, d_v1, d_v2);

  cudaMemcpy(&v3, d_v3, 3 * sizeof(Real), cudaMemcpyDeviceToHost);

  cudaFree(d_v1);
  cudaFree(d_v2);
  cudaFree(d_v3);

  std::cout << "v1 = " << v1 << std::endl;
  std::cout << "v2 = " << v2 << std::endl;
  std::cout << "v3 = " << v3 << std::endl;    

  return 0;
}
