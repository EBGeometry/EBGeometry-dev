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
    DistanceCandidate best;

    const int numTriangles = m_triangles.length();

    EBGEOMETRY_PRAGMA_SIMD
    for (int i = 0; i < numTriangles; ++i) {
      const Triangle<MetaData>& curTri = m_triangles[i];
      const DistanceCandidate   cand   = signedSquaredDistanceTriangle(curTri.getNormal(),
                                                                   curTri.getVertexPositions(),
                                                                   curTri.getVertexNormals(),
                                                                   curTri.getEdgeNormals(),
                                                                   a_point);

      compareDistanceHelper(best, cand.m_dist2, cand.m_sgn, 1);
    }

    return EBGeometry::sqrt(best.m_dist2) * best.m_sgn;
  }
} // namespace EBGeometry

#endif
