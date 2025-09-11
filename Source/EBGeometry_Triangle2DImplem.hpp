/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */
/*!
  @file   EBGeometry_Triangle2DImplem.hpp
  @brief  Implementation of EBGeometry_Triangle2DImplem.hpp
  @author Robert Marskar
*/

#ifndef EBGeometry_Triangle2DImplem
#define EBGeometry_Triangle2DImplem

// Std includes
#include <cmath>

// Our includes
#include "EBGeometry_Macros.hpp"

namespace EBGeometry {

  EBGEOMETRY_ALWAYS_INLINE
  Triangle2D::Triangle2D(const Vec3& a_normal, const Vec3 a_vertices[3]) noexcept
  {
    this->define(a_normal, a_vertices);
  }

  EBGEOMETRY_ALWAYS_INLINE
  void
  Triangle2D::define(const Vec3& a_normal, const Vec3 a_vertices[3]) noexcept
  {
    int projectDir = 0;

    for (int dir = 1; dir < 3; dir++) {
      if (std::abs(a_normal[dir]) > std::abs(a_normal[projectDir])) {
        projectDir = dir;
      }
    }

    m_xDir = 3;
    m_yDir = 0;

    for (int dir = 0; dir < 3; dir++) {
      if (dir != projectDir) {
        m_xDir = EBGeometry::min(m_xDir, dir);
        m_yDir = EBGeometry::max(m_yDir, dir);
      }
    }

    for (int i = 0; i < 3; i++) {
      m_vertices[i] = this->projectPoint(a_vertices[i]);
    }
  }

  EBGEOMETRY_ALWAYS_INLINE
  bool
  Triangle2D::isPointInside(const Vec3& a_point, const InsideOutsideAlgorithm a_algorithm) const noexcept
  {
    bool ret = false;

    EBGEOMETRY_EXPECT(m_xDir >= 0);
    EBGEOMETRY_EXPECT(m_xDir <= 2);
    EBGEOMETRY_EXPECT(m_yDir >= 0);
    EBGEOMETRY_EXPECT(m_yDir <= 2);

    switch (a_algorithm) {
    case InsideOutsideAlgorithm::SubtendedAngle: {
      ret = this->isPointInsideSubtend(a_point);

      break;
    }
    case InsideOutsideAlgorithm::CrossingNumber: {
      ret = this->isPointInsideCrossingNumber(a_point);

      break;
    }
    case InsideOutsideAlgorithm::WindingNumber: {
      ret = this->isPointInsideWindingNumber(a_point);

      break;
    }
    }

    return ret;
  }

  EBGEOMETRY_ALWAYS_INLINE
  bool
  Triangle2D::isPointInsideWindingNumber(const Vec3& a_point) const noexcept
  {
    EBGEOMETRY_EXPECT(m_xDir >= 0);
    EBGEOMETRY_EXPECT(m_xDir <= 2);
    EBGEOMETRY_EXPECT(m_yDir >= 0);
    EBGEOMETRY_EXPECT(m_yDir <= 2);

    const Vec2 projectedPoint = this->projectPoint(a_point);

    const int windingNumber = this->computeWindingNumber(projectedPoint);

    return windingNumber != 0;
  }

  EBGEOMETRY_ALWAYS_INLINE
  bool
  Triangle2D::isPointInsideCrossingNumber(const Vec3& a_point) const noexcept
  {
    EBGEOMETRY_EXPECT(m_xDir >= 0);
    EBGEOMETRY_EXPECT(m_xDir <= 2);
    EBGEOMETRY_EXPECT(m_yDir >= 0);
    EBGEOMETRY_EXPECT(m_yDir <= 2);

    const Vec2 projectedPoint = this->projectPoint(a_point);

    const int crossingNumber = this->computeCrossingNumber(projectedPoint);

    return (crossingNumber & 1);
  }

  EBGEOMETRY_ALWAYS_INLINE
  bool
  Triangle2D::isPointInsideSubtend(const Vec3& a_point) const noexcept
  {
    EBGEOMETRY_EXPECT(m_xDir >= 0);
    EBGEOMETRY_EXPECT(m_xDir <= 2);
    EBGEOMETRY_EXPECT(m_yDir >= 0);
    EBGEOMETRY_EXPECT(m_yDir <= 2);

    const Vec2 projectedPoint = this->projectPoint(a_point);

    Real sumTheta = this->computeSubtendedAngle(projectedPoint);

    sumTheta = std::abs(sumTheta) / (Real(2.0) * M_PI); // NOLINT

    return (round(sumTheta) == 1);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec2
  Triangle2D::projectPoint(const Vec3& a_point) const noexcept
  {
    EBGEOMETRY_EXPECT(m_xDir >= 0);
    EBGEOMETRY_EXPECT(m_xDir <= 2);
    EBGEOMETRY_EXPECT(m_yDir >= 0);
    EBGEOMETRY_EXPECT(m_yDir <= 2);

    return Vec2(a_point[m_xDir], a_point[m_yDir]);
  }

  EBGEOMETRY_ALWAYS_INLINE
  int
  Triangle2D::computeWindingNumber(const Vec2& a_point) const noexcept
  {
    int windingNumber = 0;

    constexpr Real zero = Real(0.0);

    auto isLeft = [](const Vec2& p0, const Vec2& p1, const Vec2& p2) {
      return (p1.x() - p0.x()) * (p2.y() - p0.y()) - (p2.x() - p0.x()) * (p1.y() - p0.y());
    };

    // Loop through all edges of the polygon
    for (int i = 0; i < 3; i++) {

      const Vec2& P  = a_point;
      const Vec2& p1 = m_vertices[i];
      const Vec2& p2 = m_vertices[(i + 1) % 3];

      const Real res = isLeft(p1, p2, P);

      if (p1.y() <= P.y()) {
        if (p2.y() > P.y() && res > zero) {
          windingNumber += 1;
        }
      }
      else {
        if (p2.y() <= P.y() && res < zero) {
          windingNumber -= 1;
        }
      }
    }

    return windingNumber;
  }

  EBGEOMETRY_ALWAYS_INLINE
  int
  Triangle2D::computeCrossingNumber(const Vec2& a_point) const noexcept
  {
    int crossingNumber = 0;

    for (int i = 0; i < 3; i++) {
      const Vec2& p1 = m_vertices[i];
      const Vec2& p2 = m_vertices[(i + 1) % 3];

      const bool upwardCrossing   = (p1.y() <= a_point.y()) && (p2.y() > a_point.y());
      const bool downwardCrossing = (p1.y() > a_point.y()) && (p2.y() <= a_point.y());

      if (upwardCrossing || downwardCrossing) {
        const Real t = (a_point.y() - p1.y()) / (p2.y() - p1.y());

        if (a_point.x() < p1.x() + t * (p2.x() - p1.x())) {
          crossingNumber += 1;
        }
      }
    }

    return crossingNumber;
  }

  EBGEOMETRY_ALWAYS_INLINE
  Real
  Triangle2D::computeSubtendedAngle(const Vec2& a_point) const noexcept
  {
    Real sumTheta = 0.0;

    for (int i = 0; i < 3; i++) {
      const Vec2 p1 = m_vertices[i] - a_point;
      const Vec2 p2 = m_vertices[(i + 1) % 3] - a_point;

      const Real theta1 = static_cast<Real>(atan2(p1.y(), p1.x()));
      const Real theta2 = static_cast<Real>(atan2(p2.y(), p2.x()));

      Real dTheta = theta2 - theta1;

      while (dTheta > M_PI) {
        dTheta -= 2.0 * M_PI;
      }
      while (dTheta < -M_PI) {
        dTheta += 2.0 * M_PI;
      }

      sumTheta += dTheta;
    }

    return sumTheta;
  }
} // namespace EBGeometry

#endif
