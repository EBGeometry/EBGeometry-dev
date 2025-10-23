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

  Triangle<int> tri(v1, v2, v3);
}

TEST_CASE("Triangle::intersects")
{
  Vec3 vertices[3];

  vertices[0] = -Vec3::unit(0);
  vertices[1] = +Vec3::unit(0);
  vertices[2] = +Vec3::unit(1);

  Triangle<int> tri(vertices[0], vertices[1], vertices[2]);

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
