// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   EBGeometry_TriangleImplem.hpp
 * @brief  Implementation of EBGeometry_Triangle.hpp
 * @author Robert Marskar
 */

#ifndef EBGEOMETRY_TRIANGLEIMPLEM_HPP
#define EBGEOMETRY_TRIANGLEIMPLEM_HPP

// Our includes
#include "EBGeometry_Triangle.hpp"
#include "EBGeometry_Macros.hpp"

namespace EBGeometry {

  template <typename MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  constexpr Triangle<MetaData>::Triangle(const Vec3& a_x1, const Vec3& a_x2, const Vec3& a_x3) noexcept
  {
    this->setVertexPositions(a_x1, a_x2, a_x3);
  }

  template <typename MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  constexpr void
  Triangle<MetaData>::setNormal(const Vec3& a_normal) noexcept
  {
    EBGEOMETRY_EXPECT(a_normal.length() > EBGeometry::Limits::eps());

    this->m_triangleNormal = a_normal / a_normal.length();
  }

  template <typename MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  constexpr void
  Triangle<MetaData>::setVertexPositions(const Vec3& a_x1, const Vec3& a_x2, const Vec3& a_x3) noexcept
  {
    m_vertexPositions[0] = a_x1;
    m_vertexPositions[1] = a_x2;
    m_vertexPositions[2] = a_x3;

    this->computeNormal();
  }

  template <typename MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  constexpr void
  Triangle<MetaData>::setVertexNormals(const Vec3& a_n1, const Vec3& a_n2, const Vec3& a_n3) noexcept
  {
    m_vertexNormals[0] = a_n1;
    m_vertexNormals[1] = a_n2;
    m_vertexNormals[2] = a_n3;
  }

  template <typename MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  constexpr void
  Triangle<MetaData>::setEdgeNormals(const Vec3& a_n1, const Vec3& a_n2, const Vec3& a_n3) noexcept
  {
    m_edgeNormals[0] = a_n1;
    m_edgeNormals[1] = a_n2;
    m_edgeNormals[2] = a_n3;
  }

  template <typename MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  constexpr void
  Triangle<MetaData>::setMetaData(const MetaData& a_metaData) noexcept
  {
    this->m_metaData = a_metaData;
  }

  template <typename MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  constexpr void
  Triangle<MetaData>::computeNormal() noexcept
  {
    const Vec3 x1x0 = m_vertexPositions[1] - m_vertexPositions[0];
    const Vec3 x2x1 = m_vertexPositions[2] - m_vertexPositions[1];

    EBGEOMETRY_EXPECT(m_vertexPositions[0] != m_vertexPositions[1]);
    EBGEOMETRY_EXPECT(m_vertexPositions[1] != m_vertexPositions[2]);
    EBGEOMETRY_EXPECT(m_vertexPositions[2] != m_vertexPositions[0]);
    EBGEOMETRY_EXPECT(x1x0.length() > EBGeometry::Limits::eps());
    EBGEOMETRY_EXPECT(x2x1.length() > EBGeometry::Limits::eps());

    m_triangleNormal = cross(x1x0, x2x1);

    EBGEOMETRY_EXPECT(m_triangleNormal.length() > EBGeometry::Limits::min());

    m_triangleNormal = m_triangleNormal / m_triangleNormal.length();

    EBGEOMETRY_EXPECT(m_triangleNormal.length() == Real(1.0));
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  constexpr Vec3&
  Triangle<MetaData>::getNormal() noexcept
  {
    return (this->m_triangleNormal);
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  constexpr const Vec3&
  Triangle<MetaData>::getNormal() const noexcept
  {
    return (this->m_triangleNormal);
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  constexpr Vec3*
  Triangle<MetaData>::getVertexPositions() noexcept
  {
    return (this->m_vertexPositions);
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  constexpr const Vec3*
  Triangle<MetaData>::getVertexPositions() const noexcept
  {
    return (this->m_vertexPositions);
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  constexpr Vec3*
  Triangle<MetaData>::getVertexNormals() noexcept
  {
    return (this->m_vertexNormals);
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  constexpr const Vec3*
  Triangle<MetaData>::getVertexNormals() const noexcept
  {
    return (this->m_vertexNormals);
  }
  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  constexpr Vec3*
  Triangle<MetaData>::getEdgeNormals() noexcept
  {
    return (this->m_edgeNormals);
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  constexpr const Vec3*
  Triangle<MetaData>::getEdgeNormals() const noexcept
  {
    return (this->m_edgeNormals);
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  constexpr MetaData&
  Triangle<MetaData>::getMetaData() noexcept
  {
    return (this->m_metaData);
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  constexpr const MetaData&
  Triangle<MetaData>::getMetaData() const noexcept
  {
    return (this->m_metaData);
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  constexpr bool
  Triangle<MetaData>::intersects(const Vec3& a_x0, const Vec3& a_x1) const noexcept
  {
    const Real epsilon = EBGeometry::Limits::eps();

    const Vec3 edge1 = m_vertexPositions[1] - m_vertexPositions[0];
    const Vec3 edge2 = m_vertexPositions[2] - m_vertexPositions[0];
    const Vec3 ray   = a_x1 - a_x0;

    EBGEOMETRY_EXPECT(edge1.length() > EBGeometry::Limits::eps());
    EBGEOMETRY_EXPECT(edge2.length() > EBGeometry::Limits::eps());
    EBGEOMETRY_EXPECT(ray.length() > EBGeometry::Limits::eps());

    const Real det = -dot(ray, m_triangleNormal);

    EBGEOMETRY_EXPECT(EBGeometry::abs(det) > EBGeometry ::Limits::eps());

    const Real invDet = Real(1.0) / det;

    const Vec3 AO  = a_x0 - m_vertexPositions[0];
    const Vec3 DAO = cross(AO, ray);

    const Real u = dot(edge2, DAO) * invDet;
    const Real v = -dot(edge1, DAO) * invDet;
    const Real t = dot(AO, m_triangleNormal) * invDet;

    const bool a = EBGeometry::abs(det) > epsilon;
    const bool b = (t >= 0.0) && (t <= 1.0);
    const bool c = (u >= 0.0) && (u <= 1.0);
    const bool d = (v >= 0.0) && (u + v) <= 1.0;

    return (a && b && c && d);
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  constexpr Real
  Triangle<MetaData>::value(const Vec3& a_point) const noexcept
  {
    auto nearOne = [](Real x) -> bool { return EBGeometry::abs(x - Real(1)) <= Real(1E-6); };

    // Perform extra checks in debug mode -- if any of these fail then something is uninitialized.
    EBGEOMETRY_EXPECT(nearOne(m_triangleNormal.length2()));

    for (int i = 0; i < 3; i++) {
      EBGEOMETRY_EXPECT(m_vertexPositions[i].length2() < EBGeometry::Limits::max());
      EBGEOMETRY_EXPECT(nearOne(m_vertexNormals[i].length2()));
      EBGEOMETRY_EXPECT(nearOne(m_edgeNormals[i].length2()));
    }

    // Here is a message from the past: If one wants, one can precompute v21, v32, v13
    // as well as many other quantities (e.g., v21.cross(m_triangleNormal)). This might
    // be helpful in order to speed things up a little bit.
    Real ret = EBGeometry::Limits::max();

    const Vec3 v21 = m_vertexPositions[1] - m_vertexPositions[0];
    const Vec3 v32 = m_vertexPositions[2] - m_vertexPositions[1];
    const Vec3 v13 = m_vertexPositions[0] - m_vertexPositions[2];

    EBGEOMETRY_EXPECT(v21.length2() > EBGeometry::Limits::eps());
    EBGEOMETRY_EXPECT(v32.length2() > EBGeometry::Limits::eps());
    EBGEOMETRY_EXPECT(v13.length2() > EBGeometry::Limits::eps());

    EBGEOMETRY_EXPECT(v21.length2() < EBGeometry::Limits::max());
    EBGEOMETRY_EXPECT(v32.length2() < EBGeometry::Limits::max());
    EBGEOMETRY_EXPECT(v13.length2() < EBGeometry::Limits::max());

    const Vec3 p1 = a_point - m_vertexPositions[0];
    const Vec3 p2 = a_point - m_vertexPositions[1];
    const Vec3 p3 = a_point - m_vertexPositions[2];

    const Real s0 = EBGeometry::sgn(dot(cross(v21, m_triangleNormal), p1));
    const Real s1 = EBGeometry::sgn(dot(cross(v32, m_triangleNormal), p2));
    const Real s2 = EBGeometry::sgn(dot(cross(v13, m_triangleNormal), p3));

    const Real t1 = dot(p1, v21) / dot(v21, v21);
    const Real t2 = dot(p2, v32) / dot(v32, v32);
    const Real t3 = dot(p3, v13) / dot(v13, v13);

    const Vec3 y1 = p1 - t1 * v21;
    const Vec3 y2 = p2 - t2 * v32;
    const Vec3 y3 = p3 - t3 * v13;

    // Distance to vertices
    ret = (p1.length() > EBGeometry::abs(ret)) ? ret : p1.length() * EBGeometry::sgn(m_vertexNormals[0].dot(p1));
    ret = (p2.length() > EBGeometry::abs(ret)) ? ret : p2.length() * EBGeometry::sgn(m_vertexNormals[1].dot(p2));
    ret = (p3.length() > EBGeometry::abs(ret)) ? ret : p3.length() * EBGeometry::sgn(m_vertexNormals[2].dot(p3));

    // Distance to edges
    const Real l1 = y1.length();
    const Real l2 = y2.length();
    const Real l3 = y3.length();

    ret = (t1 > 0.0 && t1 < 1.0 && l1 < EBGeometry::abs(ret)) ? l1 * EBGeometry::sgn(dot(m_edgeNormals[0], y1)) : ret;
    ret = (t2 > 0.0 && t2 < 1.0 && l2 < EBGeometry::abs(ret)) ? l2 * EBGeometry::sgn(dot(m_edgeNormals[1], y2)) : ret;
    ret = (t3 > 0.0 && t3 < 1.0 && l3 < EBGeometry::abs(ret)) ? l3 * EBGeometry::sgn(dot(m_edgeNormals[2], y3)) : ret;

    // Point-in-triangle. s0 + s1 + s2 >= 2.0 is a point-in-polygon test.
    const bool inside = (s0 > Real(0)) & (s1 >= Real(0)) & (s2 >= Real(0));

    return inside ? dot(m_triangleNormal, p1) : ret;
  }
} // namespace EBGeometry

#endif
