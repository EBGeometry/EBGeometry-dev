// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "../../EBGeometry.hpp"

using namespace EBGeometry;

int
main()
{
  Vec3 vx1 = Vec3::zero();
  Vec3 vx2 = Vec3::unit(0);
  Vec3 vx3 = Vec3::unit(1);

  Vec3 vn1 = Vec3::unit(2);
  Vec3 vn2 = Vec3::unit(2);
  Vec3 vn3 = Vec3::unit(2);

  Vec3 en1 = Vec3::unit(2);
  Vec3 en2 = Vec3::unit(2);
  Vec3 en3 = Vec3::unit(2);

  Triangle<int> tri(vx1, vx2, vx3, vn1, vn2, vn3, en1, en2, en3);

  Vec3 p = 4 * Vec3::one();

  Real d = tri.value(p);

  std::cout << d << std::endl;

  return 0;
}
