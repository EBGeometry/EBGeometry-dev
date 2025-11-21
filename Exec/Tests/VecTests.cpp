// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "EBGeometry.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using namespace EBGeometry;

TEST_CASE("Vec3_BasisVectors")
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

TEST_CASE("Vec3_Arithmetic")
{
  const Vec3 A = 0 * Vec3::one();
  const Vec3 B = 1 * Vec3::one();
  const Vec3 C = 2 * Vec3::one();
  const Vec3 D = 3 * Vec3::one();
  const Vec3 E = 4 * Vec3::one();

  CHECK(A + B == B);
  CHECK(B + B == C);
  CHECK(C * C == E);
  CHECK(E - D == B);
}

TEST_CASE("Vec3_Comparison")
{
  const Vec3 A = Vec3::zero();
  const Vec3 B = Vec3::one();
  const Vec3 C = Vec3::max();
  const Vec3 D = Vec3::min();
  const Vec3 E = Vec3::lowest();

  CHECK(A < B);
  CHECK(B > A);
  CHECK(B != A);
  CHECK(B == B);
  CHECK(C > B);
  CHECK(D > A);
  CHECK(E < A);
}
