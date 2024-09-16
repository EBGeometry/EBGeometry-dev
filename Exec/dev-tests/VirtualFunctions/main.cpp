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

  //  ImplicitFunction* plane_host  = new PlaneSDF(Vec3::one(), Vec3::unit(2));
  ImplicitFunction* sphere_host = new SphereSDF(Vec3::one(), 1.23456);
  ImplicitFunction* box_host    = new BoxSDF(-Vec3::one(), Vec3::one());
  ImplicitFunction* union_host  = new UnionIF(sphere_host, box_host);

  PlaneSDFFactory planeFactory(point_device, point_device);
  ImplicitFunction* plane_host = planeFactory.buildOnHost();
  ImplicitFunction** plane_device = (ImplicitFunction**) planeFactory.buildOnDevice();  

  //  ImplicitFunction** plane_device  = (GPUPointer<ImplicitFunction>)plane_host->putOnGPU();
  ImplicitFunction** sphere_device = (GPUPointer<ImplicitFunction>)sphere_host->putOnGPU();
  ImplicitFunction** box_device    = (GPUPointer<ImplicitFunction>)box_host->putOnGPU();
  //  auto union_device  = (GPUPointer<ImplicitFunction>)union_host->putOnGPU();

  UnionIFFactory factory(sphere_device, box_device);
  //  UnionIFFactory factory2(&sphere_host, &box_host);
  auto union_device = (GPUPointer<ImplicitFunction>)factory.buildOnDevice();
  //  auto union_device2 = (ImplicitFunction*) factory2.buildOnHost();

  cudaDeviceSynchronize();

  // Print plane value
  evalImplicitFunction<<<1, 1>>>(value_device, plane_device, point_device);
  cudaMemcpy(&value_host, value_device, sizeof(Real), cudaMemcpyDeviceToHost);
  std::cout << "plane value = " << value_host << "\n";

  // Print sphere value
  evalImplicitFunction<<<1, 1>>>(value_device, sphere_device, point_device);
  cudaMemcpy(&value_host, value_device, sizeof(Real), cudaMemcpyDeviceToHost);
  std::cout << "sphere value = " << value_host << "\n";

  // Print box value
  evalImplicitFunction<<<1, 1>>>(value_device, box_device, point_device);
  cudaMemcpy(&value_host, value_device, sizeof(Real), cudaMemcpyDeviceToHost);
  std::cout << "box value = " << value_host << "\n";

  // Print union value
  evalImplicitFunction<<<1, 1>>>(value_device, union_device, point_device);
  cudaMemcpy(&value_host, value_device, sizeof(Real), cudaMemcpyDeviceToHost);
  std::cout << "union value = " << value_host << "\n";

  return 0;
}
