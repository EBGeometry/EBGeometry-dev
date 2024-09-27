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

// Our includes
#include "EBGeometry_Macros.hpp"
#include "EBGeometry_DCEL_Polygon2D.hpp"

namespace EBGeometry {
  namespace DCEL {

    inline Polygon2D::Polygon2D(const Vec3& a_normal, const int a_numVertices, const Vec3* const a_points) noexcept
    {
      this->define(a_normal, a_numVertices, a_points);
    }

    inline Polygon2D::~Polygon2D() noexcept
    {
      delete[] m_points;
    }

    inline bool
    Polygon2D::isPointInside(const Vec3& a_point, const InsideOutsideAlgorithm a_algorithm) const noexcept
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

    inline bool
    Polygon2D::isPointInsidePolygonWindingNumber(const Vec3& a_point) const noexcept
    {}

    inline bool
    Polygon2D::isPointInsidePolygonSubtend(const Vec3& a_point) const noexcept
    {
      return false;
    }

    inline bool
    Polygon2D::isPointInsidePolygonCrossingNumber(const Vec3& a_point) const noexcept
    {
      return false;
    }

    inline Vec2
    Polygon2D::projectPoint(const Vec3& a_point) const noexcept
    {
      return Vec2(a_point[m_xDir], a_point[m_yDir]);
    }

    inline void
    Polygon2D::define(const Vec3& a_normal, const int a_numPoints, const Vec3* const a_points) noexcept
    {
      EBGEOMETRY_EXPECT(a_numPoints >= 3);
      EBGEOMETRY_EXPECT(a_points != nullptr);

      int ignoreDir = 0;

      for (int dir = 1; dir < 3; dir++) {
        if (std::abs(a_normal[dir] > std::abs(a_normal[ignoreDir]))) {
          ignoreDir = dir;
        }
      }

      m_xDir = 3;
      m_yDir = 0;

      for (int dir = 0; dir < 3; dir++) {
        if (dir != ignoreDir) {
          m_xDir = EBGeometry::min(m_xDir, dir);
          m_yDir = EBGeometry::max(m_yDir, dir);
        }
      }

      m_numPoints = a_numPoints;
      m_points    = new Vec2[m_numPoints];

      for (int i = 0; i < a_numPoints; i++) {
        m_points[i] = this->projectPoint(a_points[i]);
      }
    }

    inline int
    Polygon2D::computeWindingNumber(const Vec2& a_point) const noexcept
    {
      int wn = 0; // the  winding number counter

      auto isLeft = [](const Vec2& P0, const Vec2& P1, const Vec2& P2) {
        return (P1.x() - P0.x()) * (P2.y() - P0.y()) - (P2.x() - P0.x()) * (P1.y() - P0.y());
      };

      // loop through all edges of the polygon
      for (int i = 0; i < m_numPoints; i++) { // edge from V[i] to  V[i+1]

        const Vec2& P  = a_point;
        const Vec2& P1 = m_points[i];
        const Vec2& P2 = m_points[(i + 1) % m_numPoints];

        const Real res = isLeft(P1, P2, P);

        if (P1.y() <= P.y()) { // start y <= P.y
          if (P2.y() > P.y())  // an upward crossing
            if (res > 0.)      // P left of  edge
              ++wn;            // have  a valid up intersect
        }
        else {                 // start y > P.y (no test needed)
          if (P2.y() <= P.y()) // a downward crossing
            if (res < 0.)      // P right of  edge
              --wn;            // have  a valid down intersect
        }
      }

      return wn;
    }

    inline int
    Polygon2D::computeCrossingNumber(const Vec2& a_point) const noexcept
    {
      return 0;
    }

    inline Real
    Polygon2D::computeSubtendedAngle(const Vec2& a_point) const noexcept
    {
      return 0.0;
    }
  } // namespace DCEL
} // namespace EBGeometry

#endif
