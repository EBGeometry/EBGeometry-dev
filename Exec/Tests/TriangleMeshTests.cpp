#include "EBGeometry.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using namespace EBGeometry;

using MetaData = int;

TEST_CASE("TriangleMesh_Default_Constructor")
{
  TriangleMesh<MetaData> triMesh;

  CHECK(triMesh.getNumberOfTriangles() == -1);
  CHECK((triMesh.getTriangles() == nullptr));
}
