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

  PlaneSDF* plane_host   = nullptr;
  PlaneSDF* plane_device = nullptr;

  allocateImplicitFunctionOnDevice(plane_device, *origin_device, *normal_device);

  evalImplicitFunction<<<1, 1>>>(value_device, plane_device, point_device);
  cudaMemcpy(&value_host, value_device, sizeof(Real), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  std::cout << value_host << std::endl;
  freeImplicitFunctionOnDevice(plane_device);

  cudaFree(origin_device);
  cudaFree(point_device);
  cudaFree(value_device);
  cudaFree(normal_device);

  return 0;
}

  //  delete plane_host;
  //  PlaneSDF* plane_device;  

  // PlaneSDFFactory  planeFactory(point_device, point_device);
  // SphereSDFFactory sphereFactory(point_device, value_device);
  // BoxSDFFactory    boxFactory(point_device, point_device);

  // HostIF<ImplicitFunction> plane_host  = planeFactory.buildOnHost();
  // HostIF<ImplicitFunction> sphere_host = sphereFactory.buildOnHost();
  // HostIF<ImplicitFunction> box_host    = boxFactory.buildOnHost();

  // DeviceIF<ImplicitFunction> plane_device  = (DeviceIF<ImplicitFunction>)planeFactory.buildOnDevice();
  // DeviceIF<ImplicitFunction> sphere_device = (DeviceIF<ImplicitFunction>)sphereFactory.buildOnDevice();
  // DeviceIF<ImplicitFunction> box_device    = (DeviceIF<ImplicitFunction>)boxFactory.buildOnDevice();

  // UnionIFFactory     factory(sphere_device, box_device);
  // DeviceIF<ImplicitFunction> union_device = (DeviceIF<ImplicitFunction>)factory.buildOnDevice();

  // cudaDeviceSynchronize();

  // // Print plane value
  // evalImplicitFunction<<<1, 1>>>(value_device, plane_device, point_device);
  // cudaMemcpy(&value_host, value_device, sizeof(Real), cudaMemcpyDeviceToHost);
  // cudaDeviceSynchronize();
  // std::cout << "plane value = " << value_host << "\n";

  // // Print sphere value
  // evalImplicitFunction<<<1, 1>>>(value_device, sphere_device, point_device);
  // cudaMemcpy(&value_host, value_device, sizeof(Real), cudaMemcpyDeviceToHost);
  // cudaDeviceSynchronize();
  // std::cout << "sphere value = " << value_host << "\n";

  // // Print box value
  // evalImplicitFunction<<<1, 1>>>(value_device, box_device, point_device);
  // cudaMemcpy(&value_host, value_device, sizeof(Real), cudaMemcpyDeviceToHost);
  // cudaDeviceSynchronize();
  // std::cout << "box value = " << value_host << "\n";

  // // Print union value
  // evalImplicitFunction<<<1, 1>>>(value_device, union_device, point_device);
  // cudaMemcpy(&value_host, value_device, sizeof(Real), cudaMemcpyDeviceToHost);
  // cudaDeviceSynchronize();
  // std::cout << "union value = " << value_host << "\n";

  // cudaFree(point_device);
  // cudaFree(value_device);
