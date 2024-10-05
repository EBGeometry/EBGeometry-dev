#include "EBGeometry.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using namespace EBGeometry;

// These tests use a hard-coded pyramid DCEL watertight mesh given by five vertices and five faces.
//
//  v0 = (-1, -1,  0)
//  v1 = (+1, -1,  0)
//  v2 = (+1, +1,  0)
//  v3 = (-1, +1,  0)
//  v4 = ( 0,  0, +1)
//
//  There are five faces (with outward normal vectors) spanned by the following vertices:
//
//  f0 = v0, v1, v2, v3
//  f1 = v0, v1, v4
//  f2 = v1, v2, v4
//  f3 = v2, v3, v4
//  f4 = v3, v0, v4

TEST_CASE("Vertex_BasisVectors")
{
  const Vec3 A = Vec3::zero();
  const Vec3 B = Vec3::one();

  for (int i = 0; i < 3; i++) {
    const Vec3 C = Vec3::unit(i);

    CHECK(A[i] == Real(0.0));
    CHECK(B[i] == Real(1.0));
    CHECK(C[i] == Real(1.0));

    for (int j = 0; j < 3; j++) {
      if (j != i) {
        CHECK(C[j] == Real(0.0));
      }
    }
  }
}
