// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <cstdlib>
#include <iostream>
#include <filesystem>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "EBGeometry.hpp"

using namespace EBGeometry;

using MetaData = int;

TEST_CASE("TriangleMesh_Default_Constructor")
{
  TriangleMesh<MetaData> triMesh;

  CHECK(triMesh.getNumberOfTriangles() == -1);
  CHECK((triMesh.getTriangles() == nullptr));
}
