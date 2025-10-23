// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   EBGeometry_TriangleCollectionImplem.hpp
 * @author Robert Marskar
 * @brief  Implementation of EBGeometry_TriangleCollection.hpp
 */

#ifndef EBGEOMETRY_TRIANGLECOLLECTIONIMPLEM_HPP
#define EBGEOMETRY_TRIANGLECOLLECTIONIMPLEM_HPP

#include "EBGeometry_TriangleCollection.hpp"

namespace EBGeometry {

  template <typename MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  constexpr TriangleCollection<MetaData, LayoutType::AoS>::TriangleCollection(
    EBGeometry::Span<const Triangle<MetaData>> a_triangles) noexcept :
    m_triangles(a_triangles)
  {}

  template <typename MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  constexpr void
  TriangleCollection<MetaData, LayoutType::AoS>::setData(
    EBGeometry::Span<const Triangle<MetaData>> a_triangles) noexcept
  {
    m_triangles = a_triangles;
  }

  template <typename MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  int
  TriangleCollection<MetaData, LayoutType::AoS>::size() const noexcept
  {
    return m_triangles.size();
  }

  template <typename MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  Real
  TriangleCollection<MetaData, LayoutType::AoS>::value(const Vec3& a_point) const noexcept
  {
    Real ret = EBGeometry::Limits::max();

    for (int i = 0; i < m_triangles.length(); ++i) {
      const Real d = m_triangles[i].value(a_point);

      ret = (EBGeometry::abs(d) < EBGeometry::abs(ret)) ? d : ret;
    }

    return ret;
  }

  template <typename MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  TriangleCollection<MetaData, LayoutType::SoA>::TriangleCollection(
    EBGeometry::Span<const Vec3>     a_triangleNormals,
    EBGeometry::Span<const Vec3>     a_firstVertexPositions,
    EBGeometry::Span<const Vec3>     a_secondVertexPositions,
    EBGeometry::Span<const Vec3>     a_thirdVertexPositions,
    EBGeometry::Span<const Vec3>     a_firstVertexNormals,
    EBGeometry::Span<const Vec3>     a_secondVertexNormals,
    EBGeometry::Span<const Vec3>     a_thirdVertexNormals,
    EBGeometry::Span<const Vec3>     a_firstEdgeNormals,
    EBGeometry::Span<const Vec3>     a_secondEdgeNormals,
    EBGeometry::Span<const Vec3>     a_thirdEdgeNormals,
    EBGeometry::Span<const MetaData> a_metaData) noexcept
  {
    this->setData(a_triangleNormals,
                  a_firstVertexPositions,
                  a_secondVertexPositions,
                  a_thirdVertexPositions,
                  a_firstVertexNormals,
                  a_secondVertexNormals,
                  a_thirdVertexNormals,
                  a_firstEdgeNormals,
                  a_secondEdgeNormals,
                  a_thirdEdgeNormals,
                  a_metaData);
  }

  template <typename MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  constexpr void
  TriangleCollection<MetaData, LayoutType::SoA>::setData(EBGeometry::Span<const Vec3>     a_triangleNormals,
                                                         EBGeometry::Span<const Vec3>     a_firstVertexPositions,
                                                         EBGeometry::Span<const Vec3>     a_secondVertexPositions,
                                                         EBGeometry::Span<const Vec3>     a_thirdVertexPositions,
                                                         EBGeometry::Span<const Vec3>     a_firstVertexNormals,
                                                         EBGeometry::Span<const Vec3>     a_secondVertexNormals,
                                                         EBGeometry::Span<const Vec3>     a_thirdVertexNormals,
                                                         EBGeometry::Span<const Vec3>     a_firstEdgeNormals,
                                                         EBGeometry::Span<const Vec3>     a_secondEdgeNormals,
                                                         EBGeometry::Span<const Vec3>     a_thirdEdgeNormals,
                                                         EBGeometry::Span<const MetaData> a_metaData) noexcept
  {
    m_triangleNormals       = a_triangleNormals;
    m_firstVertexPositions  = a_firstVertexPositions;
    m_secondVertexPositions = a_secondVertexPositions;
    m_thirdVertexPositions  = a_thirdVertexPositions;
    m_firstVertexNormals    = a_firstVertexNormals;
    m_secondVertexNormals   = a_secondVertexNormals;
    m_thirdVertexNormals    = a_thirdVertexNormals;
    m_firstEdgeNormals      = a_firstEdgeNormals;
    m_secondEdgeNormals     = a_secondEdgeNormals;
    m_thirdEdgeNormals      = a_thirdEdgeNormals;
    m_metaData              = a_metaData;

    EBGEOMETRY_EXPECT(m_firstVertexPositions.size() == m_triangleNormals.size());
    EBGEOMETRY_EXPECT(m_secondVertexPositions.size() == m_triangleNormals.size());
    EBGEOMETRY_EXPECT(m_thirdVertexPositions.size() == m_triangleNormals.size());
    EBGEOMETRY_EXPECT(m_firstVertexNormals.size() == m_triangleNormals.size());
    EBGEOMETRY_EXPECT(m_secondVertexNormals.size() == m_triangleNormals.size());
    EBGEOMETRY_EXPECT(m_thirdVertexNormals.size() == m_triangleNormals.size());
    EBGEOMETRY_EXPECT(m_firstEdgeNormals.size() == m_triangleNormals.size());
    EBGEOMETRY_EXPECT(m_secondEdgeNormals.size() == m_triangleNormals.size());
    EBGEOMETRY_EXPECT(m_thirdEdgeNormals.size() == m_triangleNormals.size());
    EBGEOMETRY_EXPECT(m_metaData.size() == m_triangleNormals.size());
  }

  template <typename MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  int
  TriangleCollection<MetaData, LayoutType::SoA>::size() const noexcept
  {
    return m_triangleNormals.size();
  }

  template <typename MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  Real
  TriangleCollection<MetaData, LayoutType::SoA>::value(const Vec3& a_point) const noexcept
  {
    Real ret = EBGeometry::Limits::max();

    const int numTriangles = m_triangleNormals.size();

    Vec3 vertexPositions[3];
    Vec3 vertexNormals[3];
    Vec3 edgeNormals[3];

    for (int i = 0; i < numTriangles; ++i) {
      vertexPositions[0] = m_firstVertexPositions[i];
      vertexPositions[1] = m_secondVertexPositions[i];
      vertexPositions[2] = m_thirdVertexPositions[i];

      vertexNormals[0] = m_firstVertexNormals[i];
      vertexNormals[1] = m_secondVertexNormals[i];
      vertexNormals[2] = m_thirdVertexNormals[i];

      edgeNormals[0] = m_firstEdgeNormals[i];
      edgeNormals[1] = m_secondEdgeNormals[i];
      edgeNormals[2] = m_thirdEdgeNormals[i];

      const Real d = signedDistanceTriangle(m_triangleNormal[i], vertexPositions, vertexNormals, edgeNormals, a_point);

      ret = (EBGeometry::abs(d) < EBGeometry::abs(ret)) ? d : ret;
    }

    return ret;
  }

  template <typename MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  Real
  TriangleCollection<MetaData, LayoutType::SoA>::signedDistanceTriangle(const Vec3& a_triangleNormal,
                                                                        const Vec3* EBGEOMETRY_RESTRICT
                                                                          a_vertexPositions,
                                                                        const Vec3* EBGEOMETRY_RESTRICT a_vertexNormals,
                                                                        const Vec3* EBGEOMETRY_RESTRICT a_edgeNormals,
                                                                        const Vec3& a_point) noexcept
  {
#ifdef EBGEOMETRY_DEBUG
    for (int i = 0; i < 3; ++i) {
      EBGEOMETRY_ALWAYS_EXPECT(EBGeometry::abs(a_triangleNormal[i]) < EBGeometry::Limits::max());
      EBGEOMETRY_ALWAYS_EXPECT(EBGeometry::abs(a_vertexPositions[i]) < EBGeometry::Limits::max());
      EBGEOMETRY_ALWAYS_EXPECT(EBGeometry::abs(a_vertexNormals[i]) < EBGeometry::Limits::max());
      EBGEOMETRY_ALWAYS_EXPECT(EBGeometry::abs(a_edgeNormals[i]) < EBGeometry::Limits::max());
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
    ret = (p1.length() > EBGeometry::abs(ret)) ? ret : p1.length() * sgn(dot(a_vertexNormals[0], p1));
    ret = (p2.length() > EBGeometry::abs(ret)) ? ret : p2.length() * sgn(dot(a_vertexNormals[1], p2));
    ret = (p3.length() > EBGeometry::abs(ret)) ? ret : p3.length() * sgn(dot(a_vertexNormals[2], p3));

    // Distance to edges
    // clang-format off
    ret = (t1 > Real(0) && t1 < Real(1) && y1.length() < EBGeometry::abs(ret)) ? y1.length() * EBGeometry::sgn(a_edgeNormals[0].dot(y1)) : ret;
    ret = (t2 > Real(0) && t2 < Real(1) && y2.length() < EBGeometry::abs(ret)) ? y2.length() * EBGeometry::sgn(a_edgeNormals[1].dot(y2)) : ret;
    ret = (t3 > Real(0) && t3 < Real(1) && y3.length() < EBGeometry::abs(ret)) ? y3.length() * EBGeometry::sgn(a_edgeNormals[2].dot(y3)) : ret;
    // clang-format on

    // Point-in-triangle: s0 + s1 + s2 >= 2.0
    return (s0 + s1 + s2 >= Real(2)) ? dot(a_triangleNormal, p1) : ret;
  }
} // namespace EBGeometry

#endif
