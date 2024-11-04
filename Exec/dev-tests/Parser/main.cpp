// Std includes
#include <stdio.h>
#include <iostream>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>

// Our includes
#include "EBGeometry.hpp"

using namespace EBGeometry;

int
main()
{
  Vec3 point  = 8 * Vec3::one();
  Vec3 normal = 8 * Vec3::one();

  DCEL::Vertex<short> vert(point, normal, 0);
  DCEL::Edge<short>   edge;

  auto soup = MeshParser::readIntoSoup<int>("../../../Meshes/Clean/PLY/ASCII/airfoil.ply");
  //  auto mesh = MeshParser::readIntoDCEL<int>("../../../Meshes/Clean/PLY/ASCII/airfoil.stl");

  EBGEOMETRY_ALWAYS_EXPECT(EBGEOMETRY_ASSERTION_FAILURES == 0);

  return EBGEOMETRY_ASSERTION_FAILURES;
}
