/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_TriangleImplem.hpp
  @brief  Implementation of EBGeometry_Triangle.hpp
  @author Robert Marskar
*/

#ifndef EBGeometry_TriangleImplem
#define EBGeometry_TriangleImplem

// Our includes
#include "EBGeometry_Triangle.hpp"
#include "EBGeometry_Macros.hpp"

namespace EBGeometry {

  template <typename MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  Triangle<MetaData>::Triangle(const Vec3 a_vertexPositions[3]) noexcept
  {
    this->setVertexPositions(a_vertexPositions);
  }

  template <typename MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  void
  Triangle<MetaData>::setNormal(const Vec3& a_normal) noexcept
  {
    this->m_triangleNormal = a_normal;
  }

  template <typename MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  void
  Triangle<MetaData>::setVertexPositions(const Vec3 a_vertexPositions[3]) noexcept
  {
    m_vertexPositions[0] = a_vertexPositions[0];
    m_vertexPositions[1] = a_vertexPositions[1];
    m_vertexPositions[2] = a_vertexPositions[2];

    this->computeNormal();
  }

  template <typename MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  void
  Triangle<MetaData>::setVertexNormals(const Vec3 a_vertexNormals[3]) noexcept
  {
    m_vertexNormals[0] = a_vertexNormals[0];
    m_vertexNormals[1] = a_vertexNormals[1];
    m_vertexNormals[2] = a_vertexNormals[2];
  }

  template <typename MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  void
  Triangle<MetaData>::setEdgeNormals(const Vec3 a_edgeNormals[3]) noexcept
  {
    m_edgeNormals[0] = a_edgeNormals[0];
    m_edgeNormals[1] = a_edgeNormals[1];
    m_edgeNormals[2] = a_edgeNormals[2];
  }

  template <typename MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  void
  Triangle<MetaData>::setMetaData(const MetaData& a_metaData) noexcept
  {
    this->m_metaData = a_metaData;
  }

  template <typename MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  void
  Triangle<MetaData>::computeNormal() noexcept
  {
    const Vec3 x1x0 = m_vertexPositions[1] - m_vertexPositions[0];
    const Vec3 x2x1 = m_vertexPositions[2] - m_vertexPositions[1];

    EBGEOMETRY_EXPECT(x1x0.length() > EBGeometry::Limits::eps());
    EBGEOMETRY_EXPECT(x2x1.length() > EBGeometry::Limits::eps());

    m_triangleNormal = cross(x1x0, x2x1);

    EBGEOMETRY_EXPECT(m_triangleNormal.length() > EBGeometry::Limits::min());

    m_triangleNormal = m_triangleNormal / m_triangleNormal.length();

    EBGEOMETRY_EXPECT(m_triangleNormal.length() == Real(1.0));
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  Vec3&
  Triangle<MetaData>::getNormal() noexcept
  {
    return (this->m_triangleNormal);
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  const Vec3&
  Triangle<MetaData>::getNormal() const noexcept
  {
    return (this->m_triangleNormal);
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  Vec3*
  Triangle<MetaData>::getVertexPositions() noexcept
  {
    return (this->m_vertexPositions);
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  const Vec3*
  Triangle<MetaData>::getVertexPositions() const noexcept
  {
    return (this->m_vertexPositions);
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  Vec3*
  Triangle<MetaData>::getVertexNormals() noexcept
  {
    return (this->m_vertexNormals);
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  const Vec3*
  Triangle<MetaData>::getVertexNormals() const noexcept
  {
    return (this->m_vertexNormals);
  }
  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  Vec3*
  Triangle<MetaData>::getEdgeNormals() noexcept
  {
    return (this->m_edgeNormals);
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  const Vec3*
  Triangle<MetaData>::getEdgeNormals() const noexcept
  {
    return (this->m_edgeNormals);
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  MetaData&
  Triangle<MetaData>::getMetaData() noexcept
  {
    return (this->m_metaData);
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  const MetaData&
  Triangle<MetaData>::getMetaData() const noexcept
  {
    return (this->m_metaData);
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  bool
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
    const Real invDet = Real(1.0) / det;

    const Vec3 AO  = a_x0 - m_vertexPositions[0];
    const Vec3 DAO = cross(AO, ray);

    const Real u = dot(edge2, DAO) * invDet;
    const Real v = -dot(edge1, DAO) * invDet;
    const Real t = dot(AO, m_triangleNormal) * invDet;

    const bool a = abs(det) > epsilon;
    const bool b = (t >= 0.0) && (t <= 1.0);
    const bool c = (u >= 0.0) && (abs(u - 1.0) >= 0.0);
    const bool d = (v >= 0.0) && (abs(u + v - 1.0) >= 0.0);

    return (a && b && c && d);
  }

  template <typename MetaData>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  Real
  Triangle<MetaData>::signedDistance(const Vec3& a_point) const noexcept
  {
    // Perform extra checks in debug mode -- if any of these fail then something is uninitialized.
#ifdef EBGEOMETRY_DEBUG
    for (int i = 0; i < 3; i++) {
      EBGEOMETRY_ALWAYS_EXPECT(abs(m_triangleNormal[i]) < EBGeometry::Limits::max());
      EBGEOMETRY_ALWAYS_EXPECT(abs(m_vertexPositions[i]) < EBGeometry::Limits::max());
      EBGEOMETRY_ALWAYS_EXPECT(abs(m_vertexNormals[i]) < EBGeometry::Limits::max());
      EBGEOMETRY_ALWAYS_EXPECT(abs(m_edgeNormals[i]) < EBGeometry::Limits::max());
    }
#endif

    // Here is a message from the past: If one wants, one can precompute v21, v32, v13
    // as well as many other quantities (e.g., v21.cross(m_triangleNormal)). This might
    // be helpful in order to speed things up a little bit.
    Real ret = EBGeometry::Limits::max();

    const Vec3 v21 = m_vertexPositions[1] - m_vertexPositions[0];
    const Vec3 v32 = m_vertexPositions[2] - m_vertexPositions[1];
    const Vec3 v13 = m_vertexPositions[0] - m_vertexPositions[2];

    const Vec3 p1 = a_point - m_vertexPositions[0];
    const Vec3 p2 = a_point - m_vertexPositions[1];
    const Vec3 p3 = a_point - m_vertexPositions[2];

    const Real s0 = sgn(dot(cross(v21, m_triangleNormal), p1));
    const Real s1 = sgn(dot(cross(v32, m_triangleNormal), p2));
    const Real s2 = sgn(dot(cross(v13, m_triangleNormal), p3));

    const Real t1 = dot(p1, v21) / dot(v21, v21);
    const Real t2 = dot(p2, v32) / dot(v32, v32);
    const Real t3 = dot(p3, v13) / dot(v13, v13);

    const Vec3 y1 = p1 - t1 * v21;
    const Vec3 y2 = p2 - t2 * v32;
    const Vec3 y3 = p3 - t3 * v13;

    // Distance to vertices
    ret = (p1.length() > abs(ret)) ? ret : p1.length() * sgn(m_vertexNormals[0].dot(p1));
    ret = (p2.length() > abs(ret)) ? ret : p2.length() * sgn(m_vertexNormals[1].dot(p2));
    ret = (p3.length() > abs(ret)) ? ret : p3.length() * sgn(m_vertexNormals[2].dot(p3));

    // Distance to edges
    ret = (t1 > 0.0 && t1 < 1.0 && y1.length() < abs(ret)) ? y1.length() * sgn(m_edgeNormals[0].dot(y1)) : ret;
    ret = (t2 > 0.0 && t2 < 1.0 && y2.length() < abs(ret)) ? y2.length() * sgn(m_edgeNormals[1].dot(y2)) : ret;
    ret = (t3 > 0.0 && t3 < 1.0 && y3.length() < abs(ret)) ? y3.length() * sgn(m_edgeNormals[2].dot(y3)) : ret;

    // Note that s0 + s1 + s2 >= 2.0 is a point-in-polygon test.
    return (s0 + s1 + s2 >= 2.0) ? m_triangleNormal.dot(p1) : ret;
  }
} // namespace EBGeometry

#endif
