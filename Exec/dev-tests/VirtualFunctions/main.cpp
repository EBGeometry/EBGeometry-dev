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
evalImplicitFunction(Real* value,  ImplicitFunction*  func,  Vec3*  point)
{
  *value = func->value(*point);
}

int
main()
{
  Vec3 origin_host = 0 * Vec3::one();
  Vec3 point_host  = 8 * Vec3::one();
  Real value_host  = 1.234567;
  Vec3 normal_host = Vec3::one();

  Vec3* origin_device;
  Vec3* point_device;
  Real* value_device;
  Vec3* normal_device;

  cudaMallocManaged((void**)&origin_device, sizeof(Vec3));
  cudaMallocManaged((void**)&point_device, sizeof(Vec3));
  cudaMallocManaged((void**)&value_device, sizeof(Real));
  cudaMallocManaged((void**)&normal_device, sizeof(Vec3));

  cudaMemcpy(origin_device, &origin_host, sizeof(Vec3), cudaMemcpyHostToDevice);
  cudaMemcpy(point_device, &point_host, sizeof(Vec3), cudaMemcpyHostToDevice);
  cudaMemcpy(value_device, &value_host, sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy(normal_device, &normal_host, sizeof(Vec3), cudaMemcpyHostToDevice);

  // PlaneSDF* plane_host   = nullptr;
  // PlaneSDF* plane_device = nullptr;

  auto plane_host   = (ImplicitFunction*)allocateImplicitFunctionOnHost<PlaneSDF>(*origin_device, *normal_device);
  auto plane_device = (ImplicitFunction*)allocateImplicitFunctionOnDevice<PlaneSDF>(*origin_device, *normal_device);

  //  allocateImplicitFunctionOnHost<PlaneSDF>(plane_host, *origin_device, *normal_device);
  //  allocateImplicitFunctionOnDevice<PlaneSDF>(plane_device, *origin_device, *normal_device);

  evalImplicitFunction<<<1, 1>>>(value_device, plane_device, point_device);
  cudaMemcpy(&value_host, value_device, sizeof(Real), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  freeImplicitFunctionOnDevice(plane_device);
  freeImplicitFunctionOnHost(plane_host);

  cudaFree(origin_device);
  cudaFree(point_device);
  cudaFree(value_device);
  cudaFree(normal_device);

  std::cout << value_host << std::endl;

  return 0;
}
