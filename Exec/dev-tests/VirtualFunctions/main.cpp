// Std includes
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Our includes
#include "EBGeometry.hpp"

using namespace EBGeometry;

__global__ void
evalImplicitFunction(Real* value, ImplicitFunction** func, Vec3* point)
{
  *value = (*func)->value(*point);
}

int
main()
{
  Vec3 point_host = 8 * Vec3::one();
  Real value_host = -1.0;

  Vec3* point_device;
  Real* value_device;

  cudaMallocManaged((void**)&point_device, sizeof(Vec3));
  cudaMallocManaged((void**)&value_device, sizeof(Real));

  cudaMemcpy(point_device, &point_host, sizeof(Vec3), cudaMemcpyHostToDevice);
  cudaMemcpy(value_device, &value_host, sizeof(Real), cudaMemcpyHostToDevice);

  PlaneSDFFactory  planeFactory(point_device, point_device);
  SphereSDFFactory sphereFactory(point_device, value_device);
  BoxSDFFactory    boxFactory(point_device, point_device);

  std::shared_ptr<ImplicitFunction> plane_host  = planeFactory.buildOnHost();
  std::shared_ptr<ImplicitFunction> sphere_host = sphereFactory.buildOnHost();
  std::shared_ptr<ImplicitFunction> box_host    = boxFactory.buildOnHost();

  ImplicitFunction** plane_device  = (ImplicitFunction**)planeFactory.buildOnDevice();
  ImplicitFunction** sphere_device = (ImplicitFunction**)sphereFactory.buildOnDevice();
  ImplicitFunction** box_device    = (ImplicitFunction**)boxFactory.buildOnDevice();

  UnionIFFactory     factory(sphere_device, box_device);
  ImplicitFunction** union_device = (ImplicitFunction**)factory.buildOnDevice();

  cudaDeviceSynchronize();

  // Print plane value
  evalImplicitFunction<<<1, 1>>>(value_device, plane_device, point_device);
  cudaMemcpy(&value_host, value_device, sizeof(Real), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  std::cout << "plane value = " << value_host << "\n";

  // Print sphere value
  evalImplicitFunction<<<1, 1>>>(value_device, sphere_device, point_device);
  cudaMemcpy(&value_host, value_device, sizeof(Real), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  std::cout << "sphere value = " << value_host << "\n";

  // Print box value
  evalImplicitFunction<<<1, 1>>>(value_device, box_device, point_device);
  cudaMemcpy(&value_host, value_device, sizeof(Real), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  std::cout << "box value = " << value_host << "\n";

  // Print union value
  evalImplicitFunction<<<1, 1>>>(value_device, union_device, point_device);
  cudaMemcpy(&value_host, value_device, sizeof(Real), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  std::cout << "union value = " << value_host << "\n";

  cudaFree(point_device);
  cudaFree(value_device);

  return 0;
}
