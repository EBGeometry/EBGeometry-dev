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
  constexpr Triangle<MetaData>::Triangle(const Vec3& a_vx1,
                                         const Vec3& a_vx2,
                                         const Vec3& a_vx3,
                                         const Vec3& a_vn1,
                                         const Vec3& a_vn2,
                                         const Vec3& a_vn3,
                                         const Vec3& a_en1,
                                         const Vec3& a_en2,
                                         const Vec3& a_en3) noexcept
  {
    this->setVertexPositions(a_vx1, a_vx2, a_vx3);
    this->setVertexNormals(a_vn1, a_vn2, a_vn3);
    this->setEdgeNormals(a_en1, a_en2, a_en3);
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
  Triangle<MetaData>::setVertexPositions(const Vec3& a_vx1, const Vec3& a_vx2, const Vec3& a_vx3) noexcept
  {
    m_vertexPositions[0] = a_vx1;
    m_vertexPositions[1] = a_vx2;
    m_vertexPositions[2] = a_vx3;

    this->computeNormal();
  }

  template <typename MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  constexpr void
  Triangle<MetaData>::setVertexNormals(const Vec3& a_vn1, const Vec3& a_vn2, const Vec3& a_vn3) noexcept
  {
    m_vertexNormals[0] = a_vn1 / a_vn1.length();
    m_vertexNormals[1] = a_vn2 / a_vn2.length();
    m_vertexNormals[2] = a_vn3 / a_vn3.length();
  }

  template <typename MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  constexpr void
  Triangle<MetaData>::setEdgeNormals(const Vec3& a_en1, const Vec3& a_en2, const Vec3& a_en3) noexcept
  {
    m_edgeNormals[0] = a_en1 / a_en1.length();
    m_edgeNormals[1] = a_en2 / a_en2.length();
    m_edgeNormals[2] = a_en3 / a_en3.length();
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

    EBGEOMETRY_EXPECT(nearOne(m_triangleNormal.length()));
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

    const Real det    = -dot(ray, m_triangleNormal);
    const Real absDet = EBGeometry::abs(det);

    EBGEOMETRY_EXPECT(absDet > EBGeometry::Limits::eps());

    const Real invDet = Real(1.0) / det;

    const Vec3 AO  = a_x0 - m_vertexPositions[0];
    const Vec3 DAO = cross(AO, ray);

    const Real u = dot(edge2, DAO) * invDet;
    const Real v = -dot(edge1, DAO) * invDet;
    const Real t = dot(AO, m_triangleNormal) * invDet;

    const bool a = absDet > epsilon;
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

    return EBGeometry::sqrt(d.m_dist2) * d.m_sgn;
  }

  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  void
  compareDistanceHelper(DistanceCandidate& a_ret, Real a_curAbs, int a_curSgn, bool a_mask) noexcept
  {
    const bool better = a_mask & (a_curAbs < a_ret.m_dist2);

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
    // Return candidate.
    DistanceCandidate ret;

    // Sanity checks on input normal vectors.
    EBGEOMETRY_EXPECT(EBGeometry::nearOne(a_triangleNormal.length2()));
    EBGEOMETRY_EXPECT(EBGeometry::nearOne(a_vertexNormals[0].length2()));
    EBGEOMETRY_EXPECT(EBGeometry::nearOne(a_vertexNormals[1].length2()));
    EBGEOMETRY_EXPECT(EBGeometry::nearOne(a_vertexNormals[2].length2()));
    EBGEOMETRY_EXPECT(EBGeometry::nearOne(a_edgeNormals[0].length2()));
    EBGEOMETRY_EXPECT(EBGeometry::nearOne(a_edgeNormals[1].length2()));
    EBGEOMETRY_EXPECT(EBGeometry::nearOne(a_edgeNormals[2].length2()));

    constexpr Real eps = EBGeometry::Limits::eps();

    const Vec3 v21 = a_vertexPositions[1] - a_vertexPositions[0];
    const Vec3 v32 = a_vertexPositions[2] - a_vertexPositions[1];
    const Vec3 v13 = a_vertexPositions[0] - a_vertexPositions[2];

    // Sanity checks for degenerate vertices.
    EBGEOMETRY_EXPECT(v21.length2() > eps);
    EBGEOMETRY_EXPECT(v32.length2() > eps);
    EBGEOMETRY_EXPECT(v13.length2() > eps);

    // Sanity checks for unbound vertices.
    EBGEOMETRY_EXPECT(v21.length2() < EBGeometry::Limits::max());
    EBGEOMETRY_EXPECT(v32.length2() < EBGeometry::Limits::max());
    EBGEOMETRY_EXPECT(v13.length2() < EBGeometry::Limits::max());

    // Sanity checks for potentially inconsistent triangle orientations.
    EBGEOMETRY_EXPECT(dot(a_triangleNormal, cross(v21, -v32)) > 0.0);

    // Distance to vertices.
    const Vec3 p1 = a_point - a_vertexPositions[0];
    const Vec3 p2 = a_point - a_vertexPositions[1];
    const Vec3 p3 = a_point - a_vertexPositions[2];

    // Edge lengths and parametrizations of edges
    const Real d21 = dot(v21, v21);
    const Real d32 = dot(v32, v32);
    const Real d13 = dot(v13, v13);

    const Real t1 = dot(p1, v21) / (d21 + eps);
    const Real t2 = dot(p2, v32) / (d32 + eps);
    const Real t3 = dot(p3, v13) / (d13 + eps);

    const Vec3 y1 = p1 - t1 * v21;
    const Vec3 y2 = p2 - t2 * v32;
    const Vec3 y3 = p3 - t3 * v13;

    // Distances to vertices, edges, and triangle plane
    const Real p1d2 = p1.length2();
    const Real p2d2 = p2.length2();
    const Real p3d2 = p3.length2();

    const Real y1d2 = y1.length2();
    const Real y2d2 = y2.length2();
    const Real y3d2 = y3.length2();

    const Real trid  = dot(a_triangleNormal, p1);
    const Real trid2 = trid * trid;

    // Sign to vertices, edges, and triangle plane
    const int sgnV1 = EBGeometry::sgn(dot(a_vertexNormals[0], p1));
    const int sgnV2 = EBGeometry::sgn(dot(a_vertexNormals[1], p2));
    const int sgnV3 = EBGeometry::sgn(dot(a_vertexNormals[2], p3));

    const int sgnE1 = EBGeometry::sgn(dot(a_edgeNormals[0], y1));
    const int sgnE2 = EBGeometry::sgn(dot(a_edgeNormals[1], y2));
    const int sgnE3 = EBGeometry::sgn(dot(a_edgeNormals[2], y3));

    const int sgnTri = EBGeometry::sgn(trid);

    // Test if the projected point lies inside the triangle.
    const bool insideEdge0 = dot(cross(v21, a_triangleNormal), p1) <= eps;
    const bool insideEdge1 = dot(cross(v32, a_triangleNormal), p2) <= eps;
    const bool insideEdge2 = dot(cross(v13, a_triangleNormal), p3) <= eps;
    const bool okEdge1     = d21 > eps;
    const bool okEdge2     = d32 > eps;
    const bool okEdge3     = d13 > eps;

    // Masks for distance helper
    const bool maskV1  = true;
    const bool maskV2  = true;
    const bool maskV3  = true;
    const bool maskE1  = okEdge1 & (t1 > Real(0)) & (t1 < Real(1));
    const bool maskE2  = okEdge2 & (t2 > Real(0)) & (t2 < Real(1));
    const bool maskE3  = okEdge3 & (t3 > Real(0)) & (t3 < Real(1));
    const bool maskTri = insideEdge0 & insideEdge1 & insideEdge2;

    // Distances, signs, and masks
    Real dist2[7] = {p1d2, p2d2, p3d2, y1d2, y2d2, y3d2, trid2};
    int  sgn[7]   = {sgnV1, sgnV2, sgnV3, sgnE1, sgnE2, sgnE3, sgnTri};
    bool mask[7]  = {maskV1, maskV2, maskV3, maskE1, maskE2, maskE3, maskTri};

    compareDistanceHelper(ret, dist2[0], sgn[0], mask[0]);
    compareDistanceHelper(ret, dist2[1], sgn[1], mask[1]);
    compareDistanceHelper(ret, dist2[2], sgn[2], mask[2]);
    compareDistanceHelper(ret, dist2[3], sgn[3], mask[3]);
    compareDistanceHelper(ret, dist2[4], sgn[4], mask[4]);
    compareDistanceHelper(ret, dist2[5], sgn[5], mask[5]);
    compareDistanceHelper(ret, dist2[6], sgn[6], mask[6]);

    return ret;
  }
} // namespace EBGeometry

#endif
