#include "EBGeometry.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using namespace EBGeometry;

TEST_CASE("Triangle_Constructors")
{
  Vec3 vertices[3];

  vertices[0] = -Vec3::unit(0);
  vertices[1] = Vec3::unit(0);
  vertices[2] = Vec3::unit(1);

  Triangle<int> tri(vertices);
}

TEST_CASE("Triangle::intersects")
{
  Vec3 vertices[3];

  vertices[0] = -Vec3::unit(0);
  vertices[1] = Vec3::unit(0);
  vertices[2] = Vec3::unit(1);

  Triangle<int> tri(vertices);

  // Lines passes through center of triangle
  {
    const Vec3 x0 = Vec3::unit(2);
    const Vec3 x1 = -Vec3::unit(2);
    CHECK(tri.intersects(x0, x1));
  }
  {
    const Vec3 x0 = Vec3::unit(2) + Vec3::unit(0);
    const Vec3 x1 = -Vec3::unit(2) - Vec3::unit(0);
    CHECK(tri.intersects(x0, x1));
  }
  {
    const Vec3 x0 = Vec3::unit(2) + Vec3::unit(1);
    const Vec3 x1 = -Vec3::unit(2) - Vec3::unit(1);
    CHECK(tri.intersects(x0, x1));
  }

  // Lines that are parallel to triangle plane
  {
    const Vec3 x0 = Vec3::unit(2) - Vec3::unit(1);
    const Vec3 x1 = Vec3::unit(2) + Vec3::unit(1);
    CHECK(!(tri.intersects(x0, x1)));
  }
  {
    const Vec3 x0 = Vec3::unit(2) - Vec3::unit(0);
    const Vec3 x1 = Vec3::unit(2) + Vec3::unit(0);
    CHECK(!(tri.intersects(x0, x1)));
  }

  // Lines pass through triangle vertex
  {
    const Vec3 x0 = Vec3::unit(2) - Vec3::unit(0);
    const Vec3 x1 = -Vec3::unit(2) - Vec3::unit(0);
    CHECK(!(tri.intersects(x0, x1)));
  }
  {
    const Vec3 x0 = Vec3::unit(2) + Vec3::unit(0);
    const Vec3 x1 = -Vec3::unit(2) + Vec3::unit(0);
    CHECK(!(tri.intersects(x0, x1)));
  }
  {
    const Vec3 x0 = Vec3::unit(2) + Vec3::unit(1);
    const Vec3 x1 = -Vec3::unit(2) + Vec3::unit(1);
    CHECK(!(tri.intersects(x0, x1)));
  }

  {
    const Vec3 x0 = Vec3::unit(2) + 0.9 * Vec3::unit(1);
    const Vec3 x1 = -Vec3::unit(2) + 0.9 * Vec3::unit(1);
    CHECK((tri.intersects(x0, x1)));
  }
}
