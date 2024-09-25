// Std includes
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Our includes
#include "EBGeometry.hpp"

using namespace EBGeometry;

int
main()
{
  Vec3 point  = 8 * Vec3::one();
  Vec3 normal = 8 * Vec3::one();

  DCEL::Vertex<short> vert(point, normal, nullptr);
  DCEL::Edge<short>   edge(nullptr, nullptr, nullptr, nullptr);

  

  return 0;
}
