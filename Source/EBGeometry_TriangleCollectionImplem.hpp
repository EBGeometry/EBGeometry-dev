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

#warning "SoA version is not finished -- needs to be optimized"

namespace EBGeometry {

  template <typename MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  constexpr TriangleCollection<MetaData, LayoutType::AoS>::TriangleCollection(EBGeometry::Span<const Triangle<MetaData>> a_triangles) noexcept :
    m_triangles(a_triangles)
  {}

  template <typename MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  constexpr void
  TriangleCollection<MetaData, LayoutType::AoS>::setData(EBGeometry::Span<const Triangle<MetaData>> a_triangles) noexcept
  {
    m_triangles = a_triangles;
  }

  template <typename MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  constexpr int
  TriangleCollection<MetaData, LayoutType::AoS>::length() const noexcept
  {
    return m_triangles.length();
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
  constexpr TriangleCollection<MetaData, LayoutType::SoA>::TriangleCollection(EBGeometry::Span<const Vec3>     a_triangleNormals,
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

    // Sanity check on input sequences
    EBGEOMETRY_EXPECT(m_firstVertexPositions.length() == m_triangleNormals.length());
    EBGEOMETRY_EXPECT(m_secondVertexPositions.length() == m_triangleNormals.length());
    EBGEOMETRY_EXPECT(m_thirdVertexPositions.length() == m_triangleNormals.length());
    EBGEOMETRY_EXPECT(m_firstVertexNormals.length() == m_triangleNormals.length());
    EBGEOMETRY_EXPECT(m_secondVertexNormals.length() == m_triangleNormals.length());
    EBGEOMETRY_EXPECT(m_thirdVertexNormals.length() == m_triangleNormals.length());
    EBGEOMETRY_EXPECT(m_firstEdgeNormals.length() == m_triangleNormals.length());
    EBGEOMETRY_EXPECT(m_secondEdgeNormals.length() == m_triangleNormals.length());
    EBGEOMETRY_EXPECT(m_thirdEdgeNormals.length() == m_triangleNormals.length());
    EBGEOMETRY_EXPECT(m_metaData.length() == m_triangleNormals.length());
  }

  template <typename MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  constexpr int
  TriangleCollection<MetaData, LayoutType::SoA>::length() const noexcept
  {
    return m_triangleNormals.length();
  }

  template <typename MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  Real
  TriangleCollection<MetaData, LayoutType::SoA>::value(const Vec3& a_point) const noexcept
  {
    const int numTriangles = m_triangleNormals.length();

    Real retAbs = EBGeometry::Limits::max();
    int  retSgn = 1;

    EBGEOMETRY_PRAGMA_SIMD
    for (int i = 0; i < numTriangles; ++i) {
      Vec3 vp[3] = {m_firstVertexPositions[i], m_secondVertexPositions[i], m_thirdVertexPositions[i]};
      Vec3 vn[3] = {m_firstVertexNormals[i], m_secondVertexNormals[i], m_thirdVertexNormals[i]};
      Vec3 en[3] = {m_firstEdgeNormals[i], m_secondEdgeNormals[i], m_thirdEdgeNormals[i]};

      const DistanceCandidate d = EBGeometry::signedSquaredDistanceTriangle(m_triangleNormals[i], vp, vn, en, a_point);

      const bool better = (d.m_dist2 < retAbs);

      retAbs = better ? d.m_dist2 : retAbs;
      retSgn = better ? d.m_sgn : retSgn;
    }

    return (numTriangles == 0) ? EBGeometry::Limits::max() : EBGeometry::sqrt(retAbs) * retSgn;
  }

} // namespace EBGeometry

#endif
