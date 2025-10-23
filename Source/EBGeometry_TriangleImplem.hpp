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
    const DistanceCandidate d = signedSquaredDistanceTriangle(m_triangleNormal, m_vertexPositions, m_vertexNormals, m_edgeNormals, a_point);

    return sqrt(d.m_dist2) * d.m_sgn;
  }

  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  void
  compareDistanceHelper(DistanceCandidate& a_ret, Real a_curAbs, int a_curSgn, bool a_mask) noexcept
  {
    const bool better = a_mask && (a_curAbs < a_ret.m_dist2);

    a_ret.m_dist2 = better ? a_curAbs : a_ret.m_dist2;
    a_ret.m_sgn   = better ? a_curSgn : a_ret.m_sgn;
  }

  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  DistanceCandidate
  signedSquaredDistanceTriangle(const Vec3&                     a_triangleNormal,
                                const Vec3* EBGEOMETRY_RESTRICT a_vertexPositions,
                                const Vec3* EBGEOMETRY_RESTRICT a_vertexNormals,
                                const Vec3* EBGEOMETRY_RESTRICT a_edgeNormals,
                                const Vec3&                     a_point) noexcept

  {
    EBGEOMETRY_EXPECT(EBGeometry::nearOne(a_triangleNormal.length2()));
    for (int i = 0; i < 3; ++i) {
      EBGEOMETRY_EXPECT(a_vertexPositions[i].length2() < EBGeometry::Limits::max());
      EBGEOMETRY_EXPECT(EBGeometry::nearOne(a_vertexNormals[i].length2()));
      EBGEOMETRY_EXPECT(EBGeometry::nearOne(a_edgeNormals[i].length2()));
    }

    const Vec3 v21 = a_vertexPositions[1] - a_vertexPositions[0];
    const Vec3 v32 = a_vertexPositions[2] - a_vertexPositions[1];
    const Vec3 v13 = a_vertexPositions[0] - a_vertexPositions[2];

    EBGEOMETRY_EXPECT(v21.length2() > EBGeometry::Limits::eps());
    EBGEOMETRY_EXPECT(v32.length2() > EBGeometry::Limits::eps());
    EBGEOMETRY_EXPECT(v13.length2() > EBGeometry::Limits::eps());

    EBGEOMETRY_EXPECT(v21.length2() < EBGeometry::Limits::max());
    EBGEOMETRY_EXPECT(v32.length2() < EBGeometry::Limits::max());
    EBGEOMETRY_EXPECT(v13.length2() < EBGeometry::Limits::max());

    const Vec3 p1 = a_point - a_vertexPositions[0];
    const Vec3 p2 = a_point - a_vertexPositions[1];
    const Vec3 p3 = a_point - a_vertexPositions[2];

    const int s0 = EBGeometry::sgn(dot(cross(v21, a_triangleNormal), p1));
    const int s1 = EBGeometry::sgn(dot(cross(v32, a_triangleNormal), p2));
    const int s2 = EBGeometry::sgn(dot(cross(v13, a_triangleNormal), p3));

    const Real t1 = dot(p1, v21) / dot(v21, v21);
    const Real t2 = dot(p2, v32) / dot(v32, v32);
    const Real t3 = dot(p3, v13) / dot(v13, v13);
    const Real d  = dot(a_triangleNormal, p1);

    const Vec3 y1 = p1 - t1 * v21;
    const Vec3 y2 = p2 - t2 * v32;
    const Vec3 y3 = p3 - t3 * v13;

    // Point-in-triangle: s0 + s1 + s2 >= 2.0
    const bool inside = (s0 + s1 + s2 >= 2);

    // Return candidate.
    DistanceCandidate ret;

    // Distance to vertices
    compareDistanceHelper(ret, p1.length2(), EBGeometry::sgn(dot(a_vertexNormals[0], p1)), true);
    compareDistanceHelper(ret, p2.length2(), EBGeometry::sgn(dot(a_vertexNormals[1], p2)), true);
    compareDistanceHelper(ret, p3.length2(), EBGeometry::sgn(dot(a_vertexNormals[2], p3)), true);

    // Distance to edges
    compareDistanceHelper(ret, y1.length2(), EBGeometry::sgn(a_edgeNormals[0].dot(y1)), (t1 > 0.0 && t1 < 1.0));
    compareDistanceHelper(ret, y2.length2(), EBGeometry::sgn(a_edgeNormals[1].dot(y2)), (t2 > 0.0 && t2 < 1.0));
    compareDistanceHelper(ret, y3.length2(), EBGeometry::sgn(a_edgeNormals[2].dot(y3)), (t3 > 0.0 && t3 < 1.0));

    compareDistanceHelper(ret, d * d, EBGeometry::sgn(dot(a_triangleNormal, p1)), inside);

    return ret;
  }
} // namespace EBGeometry

#endif
