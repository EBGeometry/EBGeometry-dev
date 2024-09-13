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

template <typename F>
EBGEOMETRY_GPU_GLOBAL
void
evalPlane(Real* val,  const F* const func, const Vec3* const point) {
  *val = (*func)(*point);

  return;
}

EBGEOMETRY_GPU_GLOBAL
void
evalUnion(Real* val,  const ImplicitFunction* const f1, const ImplicitFunction* const f2, const Vec3* const point) {
  UnionIF csgUnion(f1, f2);
  
  *val = csgUnion.value(*point);

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
  //  cudaFree(d_v3);

  const auto hostPlane = PlaneSDF(-Vec3::one(), Vec3::unit(1));
  const auto hostPlane2 = PlaneSDF(Vec3::one(), Vec3::unit(2));  



  Real* value;
  cudaMalloc((void**) &value, sizeof(Real));

  //  cudaMalloc((void**) &devicePlane, sizeof(PlaneSDF));
  //  cudaMemcpy(devicePlane, &hostPlane, sizeof(PlaneSDF), cudaMemcpyHostToDevice);

  cudaMemcpy(&v3, d_v3, 3 * sizeof(Real), cudaMemcpyDeviceToHost);

  evalPlane<<<1, 1>>>(value, devicePlane, d_v3);

  Real hostValue;

  cudaMemcpy(&v1, d_v3, 3 * sizeof(Real), cudaMemcpyDeviceToHost);
  cudaMemcpy(&hostValue, value, sizeof(Real), cudaMemcpyDeviceToHost);    

  cudaFree(d_v1);
  cudaFree(d_v2);
  cudaFree(d_v3);

  //   devicePlane->freeOnGPU();

  // std::cout << "v1 = " << v1 << std::endl;
  // std::cout << "v2 = " << v2 << std::endl;
  // std::cout << "v3 = " << v3 << std::endl;
  // std::cout << "value = " << hostValue << std::endl;


  Vec3* v4;
  cudaMallocManaged((void**) &v4, sizeof(v4));

  *v4 = Vec3::one();

  std::cout << *v4 << std::endl;
  
  addNumbers<<<1,1>>>(v4, v4,v4);

  cudaDeviceSynchronize();
  
  std::cout << *v4 << std::endl;  

  Real* val;



  cudaMallocManaged((void**) &val, sizeof(Real));

  *val = 0.0;  

  evalUnion<<<1,1>>>(val, devicePlane, devicePlane2, v4);

  cudaDeviceSynchronize();

  std::cout << *v4 << "\t" << *val << "\t" << (hostPlane)(*v4) << "\t" << hostPlane2(*v4) << "\t" << std::endl;

  return 0;
}
