/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */
/*!
  @file   EBGeometry_Polygon2DImplem.hpp
  @brief  Implementation of EBGeometry_Polygon2D.hpp
  @author Robert Marskar
*/

#ifndef EBGeometry_Polygon2DImplem
#define EBGeometry_Polygon2DImplem

// Std includes
#include <cmath>

// Our includes
#include "EBGeometry_DCEL_Polygon2D.hpp"
#include "EBGeometry_Macros.hpp"

namespace EBGeometry::DCEL {

  EBGEOMETRY_ALWAYS_INLINE
  Polygon2D::Polygon2D() noexcept :
    m_xDir(-1),
    m_yDir(-1),
    m_numPoints(0),
    m_points(nullptr)
  {}

  EBGEOMETRY_ALWAYS_INLINE
  Polygon2D::Polygon2D(const Vec3& a_normal, int a_numPoints, const Vec3* a_points) noexcept
  {
    this->define(a_normal, a_numPoints, a_points);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Polygon2D::Polygon2D(const Polygon2D& a_polygon2D) noexcept :
    m_xDir(a_polygon2D.m_xDir),
    m_yDir(a_polygon2D.m_yDir),
    m_numPoints(a_polygon2D.m_numPoints)
  {
    EBGEOMETRY_EXPECT(m_numPoints >= 3);

    m_points = new Vec2[m_numPoints]; // NOLINT

    for (int i = 0; i < m_numPoints; i++) {
      m_points[i] = a_polygon2D.m_points[i];
    }
  }

  EBGEOMETRY_ALWAYS_INLINE
  Polygon2D::Polygon2D(Polygon2D&& a_polygon2D) noexcept :
    m_xDir(a_polygon2D.m_xDir),
    m_yDir(a_polygon2D.m_yDir),
    m_numPoints(a_polygon2D.m_numPoints)
  {
    EBGEOMETRY_EXPECT(m_numPoints >= 3);

    m_points = new Vec2[m_numPoints]; // NOLINT

    for (int i = 0; i < m_numPoints; i++) {
      m_points[i] = a_polygon2D.m_points[i];
    }
  }

  EBGEOMETRY_ALWAYS_INLINE
  Polygon2D::~Polygon2D() noexcept
  {
    if (m_points != nullptr && m_numPoints > 0) {
      delete[] m_points;
    }
  }

  EBGEOMETRY_ALWAYS_INLINE
  Polygon2D&
  Polygon2D::operator=(const Polygon2D& a_polygon2D) noexcept
  {
    if (this != &a_polygon2D) {
      m_xDir      = a_polygon2D.m_xDir;
      m_yDir      = a_polygon2D.m_yDir;
      m_numPoints = a_polygon2D.m_numPoints;

      EBGEOMETRY_EXPECT(m_numPoints >= 3);

      m_points = new Vec2[m_numPoints]; // NOLINT

      for (int i = 0; i < m_numPoints; i++) {
        m_points[i] = a_polygon2D.m_points[i];
      }
    }

    return *this;
  }

  EBGEOMETRY_ALWAYS_INLINE
  Polygon2D&
  Polygon2D::operator=(Polygon2D&& a_polygon2D) noexcept
  {
    if (this != &a_polygon2D) {
      m_xDir      = a_polygon2D.m_xDir;
      m_yDir      = a_polygon2D.m_yDir;
      m_numPoints = a_polygon2D.m_numPoints;

      EBGEOMETRY_EXPECT(m_numPoints >= 3);

      m_points = new Vec2[m_numPoints]; // NOLINT

      for (int i = 0; i < m_numPoints; i++) {
        m_points[i] = a_polygon2D.m_points[i];
      }
    }

    return *this;
  }

  EBGEOMETRY_ALWAYS_INLINE
  bool
  Polygon2D::isPointInside(const Vec3& a_point, InsideOutsideAlgorithm a_algorithm) const noexcept
  {
    bool ret = false;

    switch (a_algorithm) {
    case InsideOutsideAlgorithm::SubtendedAngle: {
      ret = this->isPointInsidePolygonSubtend(a_point);

      break;
    }
    case InsideOutsideAlgorithm::CrossingNumber: {
      ret = this->isPointInsidePolygonCrossingNumber(a_point);

      break;
    }
    case InsideOutsideAlgorithm::WindingNumber: {
      ret = this->isPointInsidePolygonWindingNumber(a_point);

      break;
    }
    }

    return ret;
  }

  EBGEOMETRY_ALWAYS_INLINE
  bool
  Polygon2D::isPointInsidePolygonWindingNumber(const Vec3& a_point) const noexcept
  {
    const Vec2 projectedPoint = this->projectPoint(a_point);

    const int windingNumber = this->computeWindingNumber(projectedPoint);

    return windingNumber != 0;
  }

  EBGEOMETRY_ALWAYS_INLINE
  bool
  Polygon2D::isPointInsidePolygonCrossingNumber(const Vec3& a_point) const noexcept
  {
    const Vec2 projectedPoint = this->projectPoint(a_point);

    const int crossingNumber = this->computeCrossingNumber(projectedPoint);

    return (crossingNumber & 1);
  }

  EBGEOMETRY_ALWAYS_INLINE
  bool
  Polygon2D::isPointInsidePolygonSubtend(const Vec3& a_point) const noexcept
  {
    const Vec2 projectedPoint = this->projectPoint(a_point);

    Real sumTheta = this->computeSubtendedAngle(projectedPoint);

    sumTheta = static_cast<Real>(std::abs(sumTheta) / Real(2. * M_PI));

    return (round(sumTheta) == 1);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec2
  Polygon2D::projectPoint(const Vec3& a_point) const noexcept
  {
    return Vec2(a_point[m_xDir], a_point[m_yDir]);
  }

  EBGEOMETRY_ALWAYS_INLINE
  void
  Polygon2D::define(const Vec3& a_normal, int a_numPoints, const Vec3* a_points) noexcept
  {
    EBGEOMETRY_EXPECT(a_numPoints >= 3);
    EBGEOMETRY_EXPECT(a_points != nullptr);

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

    // Add NOLINT to memory allocation. clang-tidy wants exception handling but there are
    // no exception on the GPU.
    m_numPoints = a_numPoints;
    m_points    = new Vec2[m_numPoints]; // NOLINT

    for (int i = 0; i < a_numPoints; i++) {
      m_points[i] = this->projectPoint(a_points[i]);
    }
  }

  EBGEOMETRY_ALWAYS_INLINE
  int
  Polygon2D::computeWindingNumber(const Vec2& a_point) const noexcept
  {
    int windingNumber = 0;

    constexpr Real zero = Real(0.0);

    auto isLeft = [](const Vec2& p0, const Vec2& p1, const Vec2& p2) {
      return (p1.x() - p0.x()) * (p2.y() - p0.y()) - (p2.x() - p0.x()) * (p1.y() - p0.y());
    };

    // Loop through all edges of the polygon
    for (int i = 0; i < m_numPoints; i++) {

      const Vec2& P  = a_point;
      const Vec2& p1 = m_points[i];
      const Vec2& p2 = m_points[(i + 1) % m_numPoints];

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
  Polygon2D::computeCrossingNumber(const Vec2& a_point) const noexcept
  {
    int crossingNumber = 0;

    for (int i = 0; i < m_numPoints; i++) {
      const Vec2& p1 = m_points[i];
      const Vec2& p2 = m_points[(i + 1) % m_numPoints];

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
  Polygon2D::computeSubtendedAngle(const Vec2& a_point) const noexcept
  {
    Real sumTheta = 0.0;

    for (int i = 0; i < m_numPoints; i++) {
      const Vec2 p1 = m_points[i] - a_point;
      const Vec2 p2 = m_points[(i + 1) % m_numPoints] - a_point;

      const Real theta1 = static_cast<Real>(atan2(p1.y(), p1.x()));
      const Real theta2 = static_cast<Real>(atan2(p2.y(), p2.x()));

      Real dTheta = theta2 - theta1;

      while (dTheta > M_PI) {
        dTheta -= Real(2.0 * M_PI);
      }
      while (dTheta < -M_PI) {
        dTheta += Real(2.0 * M_PI);
      }

      sumTheta += dTheta;
    }

    return sumTheta;
  }
} // namespace EBGeometry::DCEL

#endif
