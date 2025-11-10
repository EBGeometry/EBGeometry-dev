// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "EBGeometry.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using namespace EBGeometry;

TEST_CASE("Triangle_Constructors")
{
  const auto v1 = -Vec3::unit(0);
  const auto v2 = Vec3::unit(0);
  const auto v3 = Vec3::unit(1);

  // Compute triangle normal from vertices
  const Vec3 edge1 = v2 - v1;
  const Vec3 edge2 = v3 - v2;
  const Vec3 triNormal = cross(edge1, edge2) / cross(edge1, edge2).length();

  // Use triangle normal for vertex normals (simple approximation)
  const Vec3 n1 = triNormal;
  const Vec3 n2 = triNormal;
  const Vec3 n3 = triNormal;

  // Compute edge normals (perpendicular to edge, in triangle plane)
  const Vec3 e1 = cross(triNormal, edge1) / cross(triNormal, edge1).length();
  const Vec3 e2 = cross(triNormal, edge2) / cross(triNormal, edge2).length();
  const Vec3 edge3 = v1 - v3;
  const Vec3 e3 = cross(triNormal, edge3) / cross(triNormal, edge3).length();

  Triangle<int> tri(v1, v2, v3, n1, n2, n3, e1, e2, e3);
}
#if 0
TEST_CASE("Triangle::intersects")
{
  Vec3 vertices[3];

  vertices[0] = -Vec3::unit(0);
  vertices[1] = +Vec3::unit(0);
  vertices[2] = +Vec3::unit(1);

  // Compute triangle normal from vertices
  const Vec3 edge1 = vertices[1] - vertices[0];
  const Vec3 edge2 = vertices[2] - vertices[1];
  const Vec3 triNormal = cross(edge1, edge2) / cross(edge1, edge2).length();

  // Use triangle normal for vertex normals (simple approximation)
  const Vec3 n1 = triNormal;
  const Vec3 n2 = triNormal;
  const Vec3 n3 = triNormal;

  // Compute edge normals (perpendicular to edge, in triangle plane)
  const Vec3 e1 = cross(triNormal, edge1) / cross(triNormal, edge1).length();
  const Vec3 e2 = cross(triNormal, edge2) / cross(triNormal, edge2).length();
  const Vec3 edge3 = vertices[0] - vertices[2];
  const Vec3 e3 = cross(triNormal, edge3) / cross(triNormal, edge3).length();

  Triangle<int> tri(vertices[0], vertices[1], vertices[2], n1, n2, n3, e1, e2, e3);

  // Lines that are parallel to the triangle
  {
    const Vec3 x0 = Vec3::unit(2) - Vec3::unit(1);
    const Vec3 x1 = Vec3::unit(2) + Vec3::unit(1);
    CHECK(!(tri.intersects(x0, x1)));
    CHECK(!(tri.intersects(x1, x0)));
  }
  {
    const Vec3 x0 = Vec3::unit(2) - Vec3::unit(0);
    const Vec3 x1 = Vec3::unit(2) + Vec3::unit(0);
    CHECK(!(tri.intersects(x0, x1)));
    CHECK(!(tri.intersects(x1, x0)));
  }

  // Lines known to pass outside of the triangle
  {
    const Vec3 x0 = 2 * vertices[0] + Vec3::unit(2);
    const Vec3 x1 = 2 * vertices[0] - Vec3::unit(2);
    CHECK(!(tri.intersects(x0, x1)));
    CHECK(!(tri.intersects(x1, x0)));
  }

  // Line that passes through center of triangle
  {
    const Vec3 c  = (vertices[0] + vertices[1] + vertices[2]) / Real(3.0);
    const Vec3 x0 = c + Vec3::unit(2);
    const Vec3 x1 = c - Vec3::unit(2);
    CHECK(tri.intersects(x0, x1));
    CHECK(tri.intersects(x1, x0));
  }

  // Line that almost passes through center of triangle
  {
    const Vec3 c  = (vertices[0] + vertices[1] + vertices[2]) / Real(3.0);
    const Vec3 x0 = c + Vec3::unit(2);
    const Vec3 x1 = c + 1.E-6 * Vec3::unit(2);
    CHECK(!(tri.intersects(x0, x1)));
    CHECK(!(tri.intersects(x1, x0)));
  }

  // Lines that pass through triangle edges
  {
    for (int i = 0; i < 3; i++) {
      const Vec3 e0 = (vertices[i] + vertices[(i + 1) % 3]) / Real(2.0);
      const Vec3 x0 = e0 + Vec3::unit(2);
      const Vec3 x1 = e0 - Vec3::unit(2);
      CHECK((tri.intersects(x0, x1)));
      CHECK((tri.intersects(x1, x0)));
    }
  }

  // Lines that end on triangle edges.
  {
    for (int i = 0; i < 3; i++) {
      const Vec3 e0 = (vertices[i] + vertices[(i + 1) % 3]) / Real(2.0);
      const Vec3 x0 = e0 + Vec3::unit(2);
      const Vec3 x1 = e0;
      CHECK((tri.intersects(x0, x1)));
      CHECK((tri.intersects(x1, x0)));
    }
  }

  // Lines that pass through triangle vertices
  {
    for (int i = 0; i < 3; i++) {
      const Vec3 x0 = vertices[i] + Vec3::unit(2);
      const Vec3 x1 = vertices[i] - Vec3::unit(2);

      CHECK((tri.intersects(x0, x1)));
      CHECK((tri.intersects(x1, x0)));
    }
  }

  // Lines that end on a triangle vertex
  {
    for (int i = 0; i < 3; i++) {
      const Vec3 x0 = vertices[i] + Vec3::unit(2);
      const Vec3 x1 = vertices[i];

      CHECK((tri.intersects(x0, x1)));
      CHECK((tri.intersects(x1, x0)));
    }
  }
}
#endif
