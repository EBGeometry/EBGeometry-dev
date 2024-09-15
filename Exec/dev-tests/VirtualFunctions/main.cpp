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


template <typename T>
__global__
void makeImplicitFunction(ImplicitFunction** func)
{
  (*func) = new T();
}

__global__
void evalImplicitFunction(Real* value, ImplicitFunction** func, Vec3* point)
{
  *value = (*func)->value(*point);  
}


int
main()
{
  Vec3 point_host = Vec3::one();
  Real value_host = -1.0;
  
  Vec3* point_device;
  Real* value_device;

  cudaMalloc((void**)&point_device, sizeof(Vec3));
  cudaMalloc((void**)&value_device, sizeof(Real));  

  cudaMemcpy(point_device, &point_host, sizeof(Vec3), cudaMemcpyHostToDevice);
  cudaMemcpy(value_device, &value_host, sizeof(Real), cudaMemcpyHostToDevice);  

  ImplicitFunction* sphere_host = new SphereSDF(Vec3::one(), 1.23456);
  ImplicitFunction* box_host = new BoxSDF(-Vec3::one(), Vec3::one());
  ImplicitFunction* union_host = new UnionIF(sphere_host, box_host);  
  
  auto sphere_device = (GPUPointer<ImplicitFunction>) sphere_host->putOnGPU();
  auto box_device = (GPUPointer<ImplicitFunction>) box_host->putOnGPU();
  auto union_device = (GPUPointer<ImplicitFunction>) union_host->putOnGPU();  

  cudaDeviceSynchronize();

  // Print sphere value
  evalImplicitFunction<<<1,1>>>(value_device, sphere_device, point_device);
  cudaMemcpy(&value_host, value_device, sizeof(Real), cudaMemcpyDeviceToHost);      
  std::cout << "sphere value = " << value_host << "\n";

  // Print box value
  evalImplicitFunction<<<1,1>>>(value_device, box_device, point_device);
  cudaMemcpy(&value_host, value_device, sizeof(Real), cudaMemcpyDeviceToHost);      
  std::cout << "box value = " << value_host << "\n";  

  // Print union value
  evalImplicitFunction<<<1,1>>>(value_device, union_device, point_device);
  cudaMemcpy(&value_host, value_device, sizeof(Real), cudaMemcpyDeviceToHost);      
  std::cout << "union value = " << value_host << "\n";    

  return 0;
}
