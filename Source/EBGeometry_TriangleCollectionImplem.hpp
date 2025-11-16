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

#include "EBGeometry_Macros.hpp"
#include "EBGeometry_TriangleCollection.hpp"

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
    DistanceCandidate best{};

    const int numTriangles = m_triangles.length();

    EBGEOMETRY_PRAGMA_SIMD
    for (int i = 0; i < numTriangles; ++i) {
      const Triangle<MetaData>& curTri = m_triangles[i];

      const DistanceCandidate cand = signedSquaredDistanceTriangle(curTri.getNormal(),
                                                                   curTri.getVertexPositions(),
                                                                   curTri.getVertexNormals(),
                                                                   curTri.getEdgeNormals(),
                                                                   a_point);

      compareDistanceHelper(best, cand.m_dist2, cand.m_sgn, 1);
    }

    return EBGeometry::sqrt(best.m_dist2) * best.m_sgn;
  }

  template <typename MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  constexpr TriangleCollection<MetaData, LayoutType::SoA>::TriangleCollection(EBGeometry::Span<const Vec3>     a_triangleNormals,
                                                                              EBGeometry::Span<const Vec3>     a_vertexPos0,
                                                                              EBGeometry::Span<const Vec3>     a_vertexPos1,
                                                                              EBGeometry::Span<const Vec3>     a_vertexPos2,
                                                                              EBGeometry::Span<const Vec3>     a_vertexNorm0,
                                                                              EBGeometry::Span<const Vec3>     a_vertexNorm1,
                                                                              EBGeometry::Span<const Vec3>     a_vertexNorm2,
                                                                              EBGeometry::Span<const Vec3>     a_edgeNorm0,
                                                                              EBGeometry::Span<const Vec3>     a_edgeNorm1,
                                                                              EBGeometry::Span<const Vec3>     a_edgeNorm2,
                                                                              EBGeometry::Span<const MetaData> a_metadata) noexcept :
    m_triangleNormals(a_triangleNormals),
    m_numTriangles(static_cast<int>(a_triangleNormals.length()))
  {
    m_vertexPositions[0] = a_vertexPos0;
    m_vertexPositions[1] = a_vertexPos1;
    m_vertexPositions[2] = a_vertexPos2;
    m_vertexNormals[0]   = a_vertexNorm0;
    m_vertexNormals[1]   = a_vertexNorm1;
    m_vertexNormals[2]   = a_vertexNorm2;
    m_edgeNormals[0]     = a_edgeNorm0;
    m_edgeNormals[1]     = a_edgeNorm1;
    m_edgeNormals[2]     = a_edgeNorm2;
    m_metadata           = a_metadata;

    // Validate that all spans have identical length
    EBGEOMETRY_EXPECT(a_vertexPos0.length() == m_numTriangles);
    EBGEOMETRY_EXPECT(a_vertexPos1.length() == m_numTriangles);
    EBGEOMETRY_EXPECT(a_vertexPos2.length() == m_numTriangles);
    EBGEOMETRY_EXPECT(a_vertexNorm0.length() == m_numTriangles);
    EBGEOMETRY_EXPECT(a_vertexNorm1.length() == m_numTriangles);
    EBGEOMETRY_EXPECT(a_vertexNorm2.length() == m_numTriangles);
    EBGEOMETRY_EXPECT(a_edgeNorm0.length() == m_numTriangles);
    EBGEOMETRY_EXPECT(a_edgeNorm1.length() == m_numTriangles);
    EBGEOMETRY_EXPECT(a_edgeNorm2.length() == m_numTriangles);
    EBGEOMETRY_EXPECT(a_metadata.length() == m_numTriangles);
  }

  template <typename MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  constexpr void
  TriangleCollection<MetaData, LayoutType::SoA>::setData(EBGeometry::Span<const Vec3>     a_triangleNormals,
                                                         EBGeometry::Span<const Vec3>     a_vertexPos0,
                                                         EBGeometry::Span<const Vec3>     a_vertexPos1,
                                                         EBGeometry::Span<const Vec3>     a_vertexPos2,
                                                         EBGeometry::Span<const Vec3>     a_vertexNorm0,
                                                         EBGeometry::Span<const Vec3>     a_vertexNorm1,
                                                         EBGeometry::Span<const Vec3>     a_vertexNorm2,
                                                         EBGeometry::Span<const Vec3>     a_edgeNorm0,
                                                         EBGeometry::Span<const Vec3>     a_edgeNorm1,
                                                         EBGeometry::Span<const Vec3>     a_edgeNorm2,
                                                         EBGeometry::Span<const MetaData> a_metadata) noexcept
  {
    m_triangleNormals    = a_triangleNormals;
    m_vertexPositions[0] = a_vertexPos0;
    m_vertexPositions[1] = a_vertexPos1;
    m_vertexPositions[2] = a_vertexPos2;
    m_vertexNormals[0]   = a_vertexNorm0;
    m_vertexNormals[1]   = a_vertexNorm1;
    m_vertexNormals[2]   = a_vertexNorm2;
    m_edgeNormals[0]     = a_edgeNorm0;
    m_edgeNormals[1]     = a_edgeNorm1;
    m_edgeNormals[2]     = a_edgeNorm2;
    m_metadata           = a_metadata;
    m_numTriangles       = static_cast<int>(a_triangleNormals.length());

    // Validate that all spans have identical length
    EBGEOMETRY_EXPECT(a_vertexPos0.length() == m_numTriangles);
    EBGEOMETRY_EXPECT(a_vertexPos1.length() == m_numTriangles);
    EBGEOMETRY_EXPECT(a_vertexPos2.length() == m_numTriangles);
    EBGEOMETRY_EXPECT(a_vertexNorm0.length() == m_numTriangles);
    EBGEOMETRY_EXPECT(a_vertexNorm1.length() == m_numTriangles);
    EBGEOMETRY_EXPECT(a_vertexNorm2.length() == m_numTriangles);
    EBGEOMETRY_EXPECT(a_edgeNorm0.length() == m_numTriangles);
    EBGEOMETRY_EXPECT(a_edgeNorm1.length() == m_numTriangles);
    EBGEOMETRY_EXPECT(a_edgeNorm2.length() == m_numTriangles);
    EBGEOMETRY_EXPECT(a_metadata.length() == m_numTriangles);
  }

  template <typename MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  Real
  TriangleCollection<MetaData, LayoutType::SoA>::value(const Vec3& a_point) const noexcept
  {
    DistanceCandidate best{}; // assumes default ctor gives "worst" candidate

    const int numTriangles = m_numTriangles;

    EBGEOMETRY_PRAGMA_SIMD
    for (int i = 0; i < numTriangles; ++i) {
      const Vec3 vx[3] = {m_vertexPositions[0][i], m_vertexPositions[1][i], m_vertexPositions[2][i]};
      const Vec3 vn[3] = {m_vertexNormals[0][i], m_vertexNormals[1][i], m_vertexNormals[2][i]};
      const Vec3 en[3] = {m_edgeNormals[0][i], m_edgeNormals[1][i], m_edgeNormals[2][i]};

      const DistanceCandidate cand = signedSquaredDistanceTriangle(m_triangleNormals[i], vx, vn, en, a_point);

      compareDistanceHelper(best, cand.m_dist2, cand.m_sgn, 1);
    }

    return EBGeometry::sqrt(best.m_dist2) * best.m_sgn;
  }

} // namespace EBGeometry

#endif // EBGEOMETRY_TRIANGLECOLLECTIONIMPLEM_HPP
