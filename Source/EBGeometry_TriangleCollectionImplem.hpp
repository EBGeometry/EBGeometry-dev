// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   EBGeometry_TriangleCollectionImplem.hpp
 * @author Robert Marskar
 * @brief  Implementation of EBGeometry_TriangleCollection.hpp
 */

#ifndef EBGeometry_TriangleCollectionImplem
#define EBGeometry_TriangleCollectionImplem

#include "EBGeometry_TriangleCollection.hpp"

namespace EBGeometry {

  template <typename MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  TriangleCollection<MetaData, LayoutType::AoS>::TriangleCollection(const Triangle<MetaData>* a_triangles,
                                                                    int                       a_size) noexcept :
    m_triangles(a_triangles),
    m_size(a_size)
  {}

  template <typename MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  void
  TriangleCollection<MetaData, LayoutType::AoS>::setData(const Triangle<MetaData>* a_triangles, int a_size) noexcept
  {
    m_triangles = a_triangles;
    m_size      = a_size;
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  int
  TriangleCollection<MetaData, LayoutType::AoS>::size() const noexcept
  {
    return m_size;
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  const typename TriangleCollection<MetaData, LayoutType::AoS>::Triangle<MetaData>&
  TriangleCollection<MetaData, LayoutType::AoS>::operator[](int i) const noexcept
  {
    EBGEOMETRY_EXPECT(i >= 0 && i < m_size);

    return m_triangles[i];
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  Real
  TriangleCollection<MetaData, LayoutType::AoS>::signedDistance(const Vec3& a_point) const noexcept
  {
    Real ret = EBGeometry::Limits::max();

    for (int i = 0; i < m_size; ++i) {
      EBGEOMETRY_EXPECT(m_triangles != nullptr);

      const Real d = m_triangles[i].signedDistance(a_point);

      ret = (abs(d) < abs(ret)) ? d : ret;
    }

    return ret
  }

  template <typename MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  TriangleCollection<MetaData, LayoutType::SoA>::TriangleCollection(const Vec3*     a_triangleNormal,
                                                                    const Vec3*     a_vertexPositions[3],
                                                                    const Vec3*     a_vertexNormals[3],
                                                                    const Vec3*     a_edgeNormals[3],
                                                                    const MetaData* a_metaData,
                                                                    int             a_size) noexcept
  {
    setData(a_triangleNormal, a_vertexPositions, a_vertexNormals, a_edgeNormals, a_metaData, a_size);
  }

  template <typename MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  void
  TriangleCollection<MetaData, LayoutType::SoA>::setData(const Vec3*     a_triangleNormal,
                                                         const Vec3*     a_vertexPositions[3],
                                                         const Vec3*     a_vertexNormals[3],
                                                         const Vec3*     a_edgeNormals[3],
                                                         const MetaData* a_metaData,
                                                         int             a_size) noexcept
  {
    m_triangleNormal     = a_triangleNormal;
    m_vertexPositions[0] = a_vertexPositions[0];
    m_vertexPositions[1] = a_vertexPositions[1];
    m_vertexPositions[2] = a_vertexPositions[2];
    m_vertexNormals[0]   = a_vertexNormals[0];
    m_vertexNormals[1]   = a_vertexNormals[1];
    m_vertexNormals[2]   = a_vertexNormals[2];
    m_edgeNormals[0]     = a_edgeNormals[0];
    m_edgeNormals[1]     = a_edgeNormals[1];
    m_edgeNormals[2]     = a_edgeNormals[2];
    m_metaData           = a_metaData;
    m_size               = a_size;
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  int
  TriangleCollection<MetaData, LayoutType::SoA>::size() const noexcept
  {
    return m_size;
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  Real
  TriangleCollection<MetaData, LayoutType::SoA>::signedDistance(const Vec3& a_point) const noexcept
  {
    Real ret = EBGeometry::Limits::max();

    for (int i = 0; i < m_size; ++i) {
      Vec3 vertexPositions[3];
      Vec3 vertexNormals[3];
      Vec3 edgeNormals[3];

      vertexPositions[0] = m_vertexPositions[0][i];
      vertexPositions[1] = m_vertexPositions[1][i];
      vertexPositions[2] = m_vertexPositions[2][i];

      vertexNormals[0] = m_vertexNormals[0][i];
      vertexNormals[1] = m_vertexNormals[1][i];
      vertexNormals[2] = m_vertexNormals[2][i];

      edgeNormals[0] = m_edgedgeNormalsormals[0][i];
      edgeNormals[1] = m_edgedgeNormalsormals[1][i];
      edgeNormals[2] = m_edgedgeNormalsormals[2][i];

      const Real d = signedDistanceTriangle(m_triangleNormal[i], vertexPositions, vertexNormals, edgeNormals, a_point);

      ret = (abs(d) < abs(ret)) ? d : ret;
    }
    return ret;
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  Real
  TriangleCollection<MetaData, LayoutType::SoA>::signedDistanceTriangle(const Vec3& a_triangleNormal,
                                                                        const Vec3  a_vertexPositions[3],
                                                                        const Vec3  a_vertexNormals[3],
                                                                        const Vec3  a_edgeNormals[3],
                                                                        const Vec3& a_point) noexcept
  {
#ifdef EBGEOMETRY_DEBUG
    for (int i = 0; i < 3; ++i) {
      EBGEOMETRY_ALWAYS_EXPECT(abs(a_triangleNormal[i]) < EBGeometry::Limits::max());
      EBGEOMETRY_ALWAYS_EXPECT(abs(a_vertexPositions[i]) < EBGeometry::Limits::max());
      EBGEOMETRY_ALWAYS_EXPECT(abs(a_vertexNormals[i]) < EBGeometry::Limits::max());
      EBGEOMETRY_ALWAYS_EXPECT(abs(a_edgeNormals[i]) < EBGeometry::Limits::max());
    }
#endif

    Real ret = EBGeometry::Limits::max();

    const Vec3 v21 = a_vertexPositions[1] - a_vertexPositions[0];
    const Vec3 v32 = a_vertexPositions[2] - a_vertexPositions[1];
    const Vec3 v13 = a_vertexPositions[0] - a_vertexPositions[2];

    const Vec3 p1 = a_point - a_vertexPositions[0];
    const Vec3 p2 = a_point - a_vertexPositions[1];
    const Vec3 p3 = a_point - a_vertexPositions[2];

    const Real s0 = sgn(dot(cross(v21, a_triangleNormal), p1));
    const Real s1 = sgn(dot(cross(v32, a_triangleNormal), p2));
    const Real s2 = sgn(dot(cross(v13, a_triangleNormal), p3));

    const Real t1 = dot(p1, v21) / dot(v21, v21);
    const Real t2 = dot(p2, v32) / dot(v32, v32);
    const Real t3 = dot(p3, v13) / dot(v13, v13);

    const Vec3 y1 = p1 - t1 * v21;
    const Vec3 y2 = p2 - t2 * v32;
    const Vec3 y3 = p3 - t3 * v13;

    // Distance to vertices
    ret = (p1.length() > abs(ret)) ? ret : p1.length() * sgn(dot(a_vertexNormals[0], p1));
    ret = (p2.length() > abs(ret)) ? ret : p2.length() * sgn(dot(a_vertexNormals[1], p2));
    ret = (p3.length() > abs(ret)) ? ret : p3.length() * sgn(dot(a_vertexNormals[2], p3));

    // Distance to edges
    ret = (t1 > Real(0) && t1 < Real(1) && y1.length() < abs(ret)) ? y1.length() * sgn(a_edgeNormals[0].dot(y1)) : ret;
    ret = (t2 > Real(0) && t2 < Real(1) && y2.length() < abs(ret)) ? y2.length() * sgn(a_edgeNormals[1].dot(y2)) : ret;
    ret = (t3 > Real(0) && t3 < Real(1) && y3.length() < abs(ret)) ? y3.length() * sgn(a_edgeNormals[2].dot(y3)) : ret;

    // Point-in-triangle: s0 + s1 + s2 >= 2.0
    return (s0 + s1 + s2 >= Real(2)) ? dot(a_triangleNormal, p1) : ret;
  }
} // namespace EBGeometry

#endif // EBGeometry_TriangleCollectionImplem
