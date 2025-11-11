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
  constexpr TriangleCollection<MetaData, LayoutType::SoA>::TriangleCollection(EBGeometry::Span<const Vec3>     a_tn,
                                                                              EBGeometry::Span<const Vec3>     a_vx1,
                                                                              EBGeometry::Span<const Vec3>     a_vx2,
                                                                              EBGeometry::Span<const Vec3>     a_vx3,
                                                                              EBGeometry::Span<const Vec3>     a_vn1,
                                                                              EBGeometry::Span<const Vec3>     a_vn2,
                                                                              EBGeometry::Span<const Vec3>     a_vn3,
                                                                              EBGeometry::Span<const Vec3>     a_en1,
                                                                              EBGeometry::Span<const Vec3>     a_en2,
                                                                              EBGeometry::Span<const Vec3>     a_en3,
                                                                              EBGeometry::Span<const MetaData> a_metadata) noexcept :
    m_tn(a_tn),
    m_vx1(a_vx1),
    m_vx2(a_vx2),
    m_vx3(a_vx3),
    m_vn1(a_vn1),
    m_vn2(a_vn2),
    m_vn3(a_vn3),
    m_en1(a_en1),
    m_en2(a_en2),
    m_en3(a_en3),
    m_metadata(a_metadata),
    m_numTriangles(static_cast<int>(a_tn.length()))
  {}

  template <typename MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  constexpr void
  TriangleCollection<MetaData, LayoutType::SoA>::setData(EBGeometry::Span<const Vec3>     a_tn,
                                                         EBGeometry::Span<const Vec3>     a_vx1,
                                                         EBGeometry::Span<const Vec3>     a_vx2,
                                                         EBGeometry::Span<const Vec3>     a_vx3,
                                                         EBGeometry::Span<const Vec3>     a_vn1,
                                                         EBGeometry::Span<const Vec3>     a_vn2,
                                                         EBGeometry::Span<const Vec3>     a_vn3,
                                                         EBGeometry::Span<const Vec3>     a_en1,
                                                         EBGeometry::Span<const Vec3>     a_en2,
                                                         EBGeometry::Span<const Vec3>     a_en3,
                                                         EBGeometry::Span<const MetaData> a_metadata) noexcept
  {
    m_tn           = a_tn;
    m_vx1          = a_vx1;
    m_vx2          = a_vx2;
    m_vx3          = a_vx3;
    m_vn1          = a_vn1;
    m_vn2          = a_vn2;
    m_vn3          = a_vn3;
    m_en1          = a_en1;
    m_en2          = a_en2;
    m_en3          = a_en3;
    m_metadata     = a_metadata;
    m_numTriangles = static_cast<int>(a_tn.length());
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
      const Vec3 vx[3] = {m_vx1[i], m_vx2[i], m_vx3[i]};
      const Vec3 vn[3] = {m_vn1[i], m_vn2[i], m_vn3[i]};
      const Vec3 en[3] = {m_en1[i], m_en2[i], m_en3[i]};

      const DistanceCandidate cand = signedSquaredDistanceTriangle(m_tn[i], vx, vn, en, a_point);

      compareDistanceHelper(best, cand.m_dist2, cand.m_sgn, 1);
    }

    return EBGeometry::sqrt(best.m_dist2) * best.m_sgn;
  }

} // namespace EBGeometry

#endif // EBGEOMETRY_TRIANGLECOLLECTIONIMPLEM_HPP
