// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   EBGeometry_DCEL_EmbeddedFace2DImplem.hpp
 * @brief  Implementation of EBGeometry_DCEL_EmbeddedFace2D.hpp
 * @author Robert Marskar
 */

#ifndef EBGEOMETRY_DCEL_EMBEDDEDFACE2DIMPLEM_HPP
#define EBGEOMETRY_DCEL_EMBEDDEDFACE2DIMPLEM_HPP

// Std includes
#include <cmath>

// Our includes
#include "EBGeometry_DCEL_EmbeddedFace2D.hpp"
#include "EBGeometry_Macros.hpp"

namespace EBGeometry::DCEL {

  EBGEOMETRY_ALWAYS_INLINE
  EmbeddedFace2D::EmbeddedFace2D(const Vec3& a_normal, EBGeometry::Span<const Vec3> a_points) noexcept
  {
    // Note: This constructor signature changed but Face doesn't use it directly anymore.
    // Face now calls define() explicitly with its own storage.
  }

  EBGEOMETRY_ALWAYS_INLINE
  bool
  EmbeddedFace2D::isPointInside(const Vec3& a_point, InsideOutsideAlgorithm a_algorithm) const noexcept
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
  EmbeddedFace2D::isPointInsidePolygonWindingNumber(const Vec3& a_point) const noexcept
  {
    const Vec2 projectedPoint = this->projectPoint(a_point);

    const int windingNumber = this->computeWindingNumber(projectedPoint);

    return windingNumber != 0;
  }

  EBGEOMETRY_ALWAYS_INLINE
  bool
  EmbeddedFace2D::isPointInsidePolygonCrossingNumber(const Vec3& a_point) const noexcept
  {
    const Vec2 projectedPoint = this->projectPoint(a_point);

    const int crossingNumber = this->computeCrossingNumber(projectedPoint);

    return (crossingNumber & 1);
  }

  EBGEOMETRY_ALWAYS_INLINE
  bool
  EmbeddedFace2D::isPointInsidePolygonSubtend(const Vec3& a_point) const noexcept
  {
    const Vec2 projectedPoint = this->projectPoint(a_point);

    Real sumTheta = this->computeSubtendedAngle(projectedPoint);

    sumTheta = static_cast<Real>(abs(sumTheta) / Real(2. * M_PI));

    return (round(sumTheta) == 1);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec2
  EmbeddedFace2D::projectPoint(const Vec3& a_point) const noexcept
  {
    return Vec2(a_point[m_xDir], a_point[m_yDir]);
  }

  EBGEOMETRY_ALWAYS_INLINE
  void
  EmbeddedFace2D::define(const Vec3&                  a_normal,
                         EBGeometry::Span<const Vec3> a_points,
                         EBGeometry::Span<Vec2>       a_polygon2DPoints) noexcept
  {
    EBGEOMETRY_EXPECT(a_points.length() >= 3);
    EBGEOMETRY_EXPECT(a_points.data() != nullptr);
    EBGEOMETRY_EXPECT(a_polygon2DPoints.data() != nullptr);
    EBGEOMETRY_EXPECT(a_points.length() == a_polygon2DPoints.length());

    const int numPoints = a_points.length();

    // Find the direction with the largest normal component (this is the one we'll project out)
    int projectDir = 0;

    for (int dir = 1; dir < 3; dir++) {
      if (abs(a_normal[dir]) > abs(a_normal[projectDir])) {
        projectDir = dir;
      }
    }

    // Determine which two directions we'll keep for the 2D embedding
    m_xDir = 3;
    m_yDir = 0;

    for (int dir = 0; dir < 3; dir++) {
      if (dir != projectDir) {
        m_xDir = EBGeometry::min(m_xDir, dir);
        m_yDir = EBGeometry::max(m_yDir, dir);
      }
    }

    // Project all 3D vertices to 2D and store in the provided array
    for (int i = 0; i < numPoints; i++) {
      a_polygon2DPoints[i] = this->projectPoint(a_points[i]);
    }

    // Store a Span view into the array
    m_points = EBGeometry::Span<const Vec2>(a_polygon2DPoints.data(), numPoints);
  }

  EBGEOMETRY_ALWAYS_INLINE
  int
  EmbeddedFace2D::computeWindingNumber(const Vec2& a_point) const noexcept
  {
    int windingNumber = 0;

    const int numPoints = m_points.length();

    constexpr Real zero = Real(0.0);

    auto isLeft = [](const Vec2& p0, const Vec2& p1, const Vec2& p2) {
      return (p1.x() - p0.x()) * (p2.y() - p0.y()) - (p2.x() - p0.x()) * (p1.y() - p0.y());
    };

    // Loop through all edges of the polygon
    for (int i = 0; i < numPoints; i++) {

      const Vec2& P  = a_point;
      const Vec2& p1 = m_points[i];
      const Vec2& p2 = m_points[(i + 1) % numPoints];

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
  EmbeddedFace2D::computeCrossingNumber(const Vec2& a_point) const noexcept
  {
    int crossingNumber = 0;

    const int numPoints = m_points.length();

    for (int i = 0; i < numPoints; i++) {
      const Vec2& p1 = m_points[i];
      const Vec2& p2 = m_points[(i + 1) % numPoints];

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
  EmbeddedFace2D::computeSubtendedAngle(const Vec2& a_point) const noexcept
  {
    Real sumTheta = 0.0;

    const int numPoints = m_points.length();

    for (int i = 0; i < numPoints; i++) {
      const Vec2 p1 = m_points[i] - a_point;
      const Vec2 p2 = m_points[(i + 1) % numPoints] - a_point;

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
