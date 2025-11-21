// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

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

  EBGEOMETRY_EXPECT(1 == 2);
  DCEL::Vertex<short> vert(point, normal, 0);
  DCEL::Edge<short>   edge();

  return 0;
}
