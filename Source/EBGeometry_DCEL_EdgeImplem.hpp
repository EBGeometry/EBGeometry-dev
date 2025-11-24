// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   EBGeometry_DCEL_EdgeImplem.hpp
 * @brief  Implementation of EBGeometry_DCEL_Edge.hpp
 * @author Robert Marskar
 */

#ifndef EBGeometry_DCEL_EdgeImplem
#define EBGeometry_DCEL_EdgeImplem

// Our includes
#include "EBGeometry_DCEL_Edge.hpp"
#include "EBGeometry_Macros.hpp"

namespace EBGeometry::DCEL {

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  Edge<MetaData>::Edge(int a_vertex) noexcept :
    m_vertex(a_vertex)
  {}

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  Edge<MetaData>::Edge(int a_vertex, int a_previousEdge, int a_pairEdge, int a_nextEdge, int a_face) noexcept :
    m_vertex(a_vertex),
    m_previousEdge(a_previousEdge),
    m_pairEdge(a_pairEdge),
    m_nextEdge(a_nextEdge),
    m_face(a_face)
  {}

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  void
  Edge<MetaData>::setVertex(int a_vertex) noexcept
  {
    m_vertex = a_vertex;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  void
  Edge<MetaData>::setPreviousEdge(int a_previousEdge) noexcept
  {
    m_previousEdge = a_previousEdge;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  void
  Edge<MetaData>::setPairEdge(int a_pairEdge) noexcept
  {
    m_pairEdge = a_pairEdge;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  void
  Edge<MetaData>::setNextEdge(int a_nextEdge) noexcept
  {
    m_nextEdge = a_nextEdge;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  void
  Edge<MetaData>::setFace(int a_face) noexcept
  {
    m_face = a_face;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  void
  Edge<MetaData>::setMetaData(const MetaData& a_metaData) noexcept
  {
    m_metaData = a_metaData;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  void
  Edge<MetaData>::setVertexList(EBGeometry::Span<const Vertex<MetaData>> a_vertexList) noexcept
  {
    m_vertexList = a_vertexList;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  void
  Edge<MetaData>::setEdgeList(EBGeometry::Span<const Edge<MetaData>> a_edgeList) noexcept
  {
    m_edgeList = a_edgeList;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  void
  Edge<MetaData>::setFaceList(EBGeometry::Span<const Face<MetaData>> a_faceList) noexcept
  {
    m_faceList = a_faceList;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  EBGeometry::Span<const Vertex<MetaData>>
  Edge<MetaData>::getVertexList() const noexcept
  {
    return m_vertexList;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  EBGeometry::Span<const Edge<MetaData>>
  Edge<MetaData>::getEdgeList() const noexcept
  {
    return m_edgeList;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  EBGeometry::Span<const Face<MetaData>>
  Edge<MetaData>::getFaceList() const noexcept
  {
    return m_faceList;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  void
  Edge<MetaData>::setNormal(const Vec3& a_normal) noexcept
  {
    m_normal = a_normal;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  void
  Edge<MetaData>::computeNormal() noexcept
  {
    EBGEOMETRY_ALWAYS_EXPECT(m_faceList.data() != nullptr);
    EBGEOMETRY_ALWAYS_EXPECT(m_face >= 0);
    EBGEOMETRY_ALWAYS_EXPECT(m_face < m_faceList.length());

    EBGEOMETRY_ALWAYS_EXPECT(m_edgeList.data() != nullptr);
    EBGEOMETRY_ALWAYS_EXPECT(m_pairEdge >= 0);
    EBGEOMETRY_ALWAYS_EXPECT(m_pairEdge < m_edgeList.length());

    EBGEOMETRY_ALWAYS_EXPECT(m_edgeList[m_pairEdge].getFace() >= 0);
    EBGEOMETRY_ALWAYS_EXPECT(m_edgeList[m_pairEdge].getFace() < m_faceList.length());

    m_normal = Vec3::zero();

    m_normal += m_faceList[m_face].getNormal();
    m_normal += m_faceList[m_edgeList[m_pairEdge].getFace()].getNormal();

    this->normalizeNormalVector();
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  void
  Edge<MetaData>::normalizeNormalVector() noexcept
  {
    constexpr Real eps = EBGeometry::Limits::eps();

    const Real len = m_normal.length();

    EBGEOMETRY_ALWAYS_EXPECT(m_normal.length() > eps);

    m_normal = (m_normal.length() > eps) ? m_normal / len : Vec3::unit(2);
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  int
  Edge<MetaData>::getVertex() const noexcept
  {
    return m_vertex;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  int
  Edge<MetaData>::getOtherVertex() const noexcept
  {
    EBGEOMETRY_ALWAYS_EXPECT(m_edgeList.data() != nullptr);
    EBGEOMETRY_ALWAYS_EXPECT(m_nextEdge >= 0);
    EBGEOMETRY_ALWAYS_EXPECT(m_nextEdge < m_edgeList.length());

    return m_edgeList[m_nextEdge].getVertex();
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  int
  Edge<MetaData>::getPreviousEdge() const noexcept
  {
    return m_previousEdge;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  int
  Edge<MetaData>::getPairEdge() const noexcept
  {
    return m_pairEdge;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  int
  Edge<MetaData>::getNextEdge() const noexcept
  {
    return m_nextEdge;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  int
  Edge<MetaData>::getFace() const noexcept
  {
    return m_face;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  const Vec3&
  Edge<MetaData>::getNormal() const noexcept
  {
    return m_normal;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  MetaData&
  Edge<MetaData>::getMetaData() noexcept
  {
    return m_metaData;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  const MetaData&
  Edge<MetaData>::getMetaData() const noexcept
  {
    return m_metaData;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  Vec3
  Edge<MetaData>::getX2X1() const noexcept
  {
    EBGEOMETRY_EXPECT(m_vertexList.data() != nullptr);
    EBGEOMETRY_EXPECT(m_vertex >= 0);
    EBGEOMETRY_EXPECT(m_vertex < m_vertexList.length());
    EBGEOMETRY_EXPECT(this->getOtherVertex() >= 0);
    EBGEOMETRY_EXPECT(this->getOtherVertex() < m_vertexList.length());

    const auto& x1 = m_vertexList[this->getVertex()].getPosition();
    const auto& x2 = m_vertexList[this->getOtherVertex()].getPosition();

    return x2 - x1;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  Real
  Edge<MetaData>::projectPointToEdge(const Vec3& a_x0) const noexcept
  {
    EBGEOMETRY_EXPECT(m_vertexList.data() != nullptr);
    EBGEOMETRY_EXPECT(m_vertex >= 0);
    EBGEOMETRY_EXPECT(m_vertex < m_vertexList.length());

    const Vec3 p    = a_x0 - m_vertexList[m_vertex].getPosition();
    const Vec3 x2x1 = this->getX2X1();
    const Real len  = dot(x2x1, x2x1);

    EBGEOMETRY_EXPECT(len > EBGeometry::Limits::eps());

    return (len > EBGeometry::Limits::eps()) ? dot(p, x2x1) / len : 0.0;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  Real
  Edge<MetaData>::signedDistance(const Vec3& a_x0) const noexcept
  {
    EBGEOMETRY_EXPECT(m_vertexList.data() != nullptr);
    EBGEOMETRY_EXPECT(this->getVertex() >= 0);
    EBGEOMETRY_EXPECT(this->getVertex() < m_vertexList.length());
    EBGEOMETRY_EXPECT(this->getOtherVertex() >= 0);
    EBGEOMETRY_EXPECT(this->getOtherVertex() < m_vertexList.length());

    // Project point to edge
    const Real t = this->projectPointToEdge(a_x0);

    Real retval = EBGeometry::Limits::max();

    if (t <= 0.0) {
      // Closest point is the starting vertex.
      retval = m_vertexList[this->getVertex()].signedDistance(a_x0);
    }
    else if (t >= 1.0) {
      // Closest point is the end vertex.
      retval = m_vertexList[this->getOtherVertex()].signedDistance(a_x0);
    }
    else {
      // Closest point is the edge itself.
      const Vec3 x2x1      = this->getX2X1();
      const Vec3 linePoint = m_vertexList[m_vertex].getPosition() + t * x2x1;
      const Vec3 delta     = a_x0 - linePoint;
      const Real dot       = m_normal.dot(delta);

      const int sgn = (dot > 0.0) ? 1 : -1;

      retval = Real(sgn) * delta.length();
    }

    return retval;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  Real
  Edge<MetaData>::unsignedDistance2(const Vec3& a_x0) const noexcept
  {
    EBGEOMETRY_EXPECT(m_vertexList != nullptr);
    EBGEOMETRY_EXPECT(m_vertex >= 0);

    constexpr Real zero = 0.0;
    constexpr Real one  = 1.0;

    // Project point to edge and restrict to edge length.
    const auto t = std::min(std::max(zero, this->projectPointToEdge(a_x0)), one);

    // Compute distance to this edge.
    const Vec3 x2x1      = this->getX2X1();
    const Vec3 linePoint = m_vertexList[m_vertex].getPosition() + t * x2x1;
    const Vec3 delta     = a_x0 - linePoint;

    return delta.dot(delta);
  }
} // namespace EBGeometry::DCEL

#endif
