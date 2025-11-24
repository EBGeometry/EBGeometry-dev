// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   EBGeometry_DCEL_VertexImplem.hpp
 * @brief  Implementation of EBGeometry_DCEL_Vertex.hpp
 * @author Robert Marskar
 */

#ifndef EBGEOMETRY_DCEL_VERTEXIMPLEM_HPP
#define EBGEOMETRY_DCEL_VERTEXIMPLEM_HPP

// Std includes
#include <cmath>

// Our includes
#include "EBGeometry_DCEL_Vertex.hpp"
#include "EBGeometry_Macros.hpp"

namespace EBGeometry::DCEL {

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  Vertex<MetaData>::Vertex(const Vec3& a_position) noexcept :
    m_position(a_position)
  {}

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  Vertex<MetaData>::Vertex(const Vec3& a_position, const Vec3& a_normal) noexcept :
    m_position(a_position),
    m_normal(a_normal)
  {}

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  Vertex<MetaData>::Vertex(const Vec3& a_position, const Vec3& a_normal, int a_edge) noexcept :
    m_position(a_position),
    m_normal(a_normal),
    m_outgoingEdge(a_edge)
  {}

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  void
  Vertex<MetaData>::setPosition(const Vec3& a_position) noexcept
  {
    m_position = a_position;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  void
  Vertex<MetaData>::setNormal(const Vec3& a_normal) noexcept
  {
    m_normal = a_normal;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  void
  Vertex<MetaData>::setEdge(int a_edge) noexcept
  {
    m_outgoingEdge = a_edge;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  void
  Vertex<MetaData>::setMetaData(const MetaData& a_metaData) noexcept
  {
    m_metaData = a_metaData;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  void
  Vertex<MetaData>::setVertexList(EBGeometry::Span<const Vertex<MetaData>> a_vertexList) noexcept
  {
    m_vertexList = a_vertexList;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  void
  Vertex<MetaData>::setEdgeList(EBGeometry::Span<const Edge<MetaData>> a_edgeList) noexcept
  {
    m_edgeList = a_edgeList;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  void
  Vertex<MetaData>::setFaceList(EBGeometry::Span<const Face<MetaData>> a_faceList) noexcept
  {
    m_faceList = a_faceList;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  EBGeometry::Span<const Vertex<MetaData>>
  Vertex<MetaData>::getVertexList() const noexcept
  {
    return m_vertexList;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  EBGeometry::Span<const Edge<MetaData>>
  Vertex<MetaData>::getEdgeList() const noexcept
  {
    return m_edgeList;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  EBGeometry::Span<const Face<MetaData>>
  Vertex<MetaData>::getFaceList() const noexcept
  {
    return m_faceList;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  void
  Vertex<MetaData>::normalizeNormalVector() noexcept
  {
    const Real len = m_normal.length();

    EBGEOMETRY_EXPECT(len > 0.0);

    // Degenerate case: default to unit z-vector (mesh will be garbage).
    m_normal = (len > EBGeometry::Limits::eps()) ? m_normal / len : Vec3::unit(2);
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  void
  Vertex<MetaData>::computeVertexNormalAverage() noexcept
  {
    // This routine computes the normal vector using a weighted sum of all faces
    // that share this vertex.
    EBGEOMETRY_ALWAYS_EXPECT(m_outgoingEdge >= 0);
    EBGEOMETRY_ALWAYS_EXPECT(m_outgoingEdge < m_edgeList.length());

    EBGEOMETRY_ALWAYS_EXPECT(m_vertexList.data() != nullptr);
    EBGEOMETRY_ALWAYS_EXPECT(m_vertexList.length() > 0);

    EBGEOMETRY_ALWAYS_EXPECT(m_edgeList.data() != nullptr);
    EBGEOMETRY_ALWAYS_EXPECT(m_edgeList.length() > 0);

    EBGEOMETRY_ALWAYS_EXPECT(m_faceList.data() != nullptr);
    EBGEOMETRY_ALWAYS_EXPECT(m_faceList.length() > 0);

    m_normal = Vec3::zero();

    int curEdge = -1;
    int curFace = -1;

    int       iterCount = 0;
    const int maxIters  = m_edgeList.length();

    while (curEdge != m_outgoingEdge) {
      // Detect malformed DCEL structure
      EBGEOMETRY_ALWAYS_EXPECT(iterCount < maxIters);

      if (iterCount >= maxIters) {
        m_normal = Vec3::unit(2);

        break;
      }

      iterCount++;

      curEdge = (curEdge < 0) ? m_outgoingEdge : curEdge;
      EBGEOMETRY_ALWAYS_EXPECT(curEdge >= 0 && curEdge < m_edgeList.length());

      curFace = m_edgeList[curEdge].getFace();
      EBGEOMETRY_ALWAYS_EXPECT(curFace >= 0 && curFace < m_faceList.length());

      m_normal += m_faceList[curFace].getNormal();

      // Jump to the pair edge and advance so we get the outgoing edge (from this vertex) on
      // the next polygon.
      curEdge = m_edgeList[curEdge].getPairEdge();
      EBGEOMETRY_ALWAYS_EXPECT(curEdge >= 0 && curEdge < m_edgeList.length());

      curEdge = m_edgeList[curEdge].getNextEdge();
      EBGEOMETRY_ALWAYS_EXPECT(curEdge >= 0 && curEdge < m_edgeList.length());
    }

    // Ensure accumulated normal has non-zero length before normalizing
    // (prevents division by zero when opposing face normals cancel out)
    EBGEOMETRY_ALWAYS_EXPECT(m_normal.length() > EBGeometry::Limits::eps());

    this->normalizeNormalVector();
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  void
  Vertex<MetaData>::computeVertexNormalAngleWeighted() noexcept
  {
    m_normal = Vec3::zero();

    // This routine computes the normal vector using the pseudonormal algorithm by
    // Baerentzen and Aanes in "Signed distance computation using the angle
    // weighted pseudonormal" (DOI: 10.1109/TVCG.2005.49). This algorithm computes
    // an average normal vector using the normal vectors of each face connected to
    // this vertex, i.e. in the form
    //
    //    n = sum(w * n(face))/sum(w)
    //
    // where w are weights for each face. This weight is given by the subtended
    // angle of the face, which means the angle spanned by the incoming/outgoing
    // edges of the face that pass through this vertex.
    EBGEOMETRY_ALWAYS_EXPECT(m_outgoingEdge >= 0);
    EBGEOMETRY_ALWAYS_EXPECT(m_outgoingEdge < m_edgeList.length());

    EBGEOMETRY_ALWAYS_EXPECT(m_vertexList.data() != nullptr);
    EBGEOMETRY_ALWAYS_EXPECT(m_vertexList.length() > 0);

    EBGEOMETRY_ALWAYS_EXPECT(m_edgeList.data() != nullptr);
    EBGEOMETRY_ALWAYS_EXPECT(m_edgeList.length() > 0);

    EBGEOMETRY_ALWAYS_EXPECT(m_faceList.data() != nullptr);
    EBGEOMETRY_ALWAYS_EXPECT(m_faceList.length() > 0);

    int outgoingEdge = -1;
    int incomingEdge = -1;
    int faceIndex    = -1;

    int       iterCount = 0;
    const int maxIters  = m_edgeList.length();

    while (outgoingEdge != m_outgoingEdge) {
      // Detect malformed DCEL structure
      EBGEOMETRY_ALWAYS_EXPECT(iterCount < maxIters);

      if (iterCount >= maxIters) {
        m_normal = Vec3::unit(2);

        break;
      }

      iterCount++;

      // Get the incoming and outgoing edges out of the origin vertex.
      outgoingEdge = (outgoingEdge < 0) ? m_outgoingEdge : outgoingEdge;
      EBGEOMETRY_ALWAYS_EXPECT(outgoingEdge >= 0 && outgoingEdge < m_edgeList.length());

      incomingEdge = m_edgeList[outgoingEdge].getPreviousEdge();
      EBGEOMETRY_ALWAYS_EXPECT(incomingEdge >= 0 && incomingEdge < m_edgeList.length());

      faceIndex = m_edgeList[outgoingEdge].getFace();
      EBGEOMETRY_ALWAYS_EXPECT(faceIndex >= 0 && faceIndex < m_faceList.length());

      EBGEOMETRY_ALWAYS_EXPECT(outgoingEdge != incomingEdge);

      // Vertices are named v0,v1,v2:
      // v0 = Origin vertex of incoming edge
      // v1 = this vertex
      // v2 = End vertex of outgoing edge.
      const int v0 = m_edgeList[incomingEdge].getVertex();
      EBGEOMETRY_ALWAYS_EXPECT(v0 >= 0 && v0 < m_vertexList.length());

      const int v2 = m_edgeList[outgoingEdge].getOtherVertex();
      EBGEOMETRY_ALWAYS_EXPECT(v2 >= 0 && v2 < m_vertexList.length());
      EBGEOMETRY_ALWAYS_EXPECT(v0 != v2);

      const Vec3& x0 = m_vertexList[v0].getPosition();
      const Vec3& x1 = m_position;
      const Vec3& x2 = m_vertexList[v2].getPosition();

      EBGEOMETRY_ALWAYS_EXPECT(x0 != x1);
      EBGEOMETRY_ALWAYS_EXPECT(x1 != x2);
      EBGEOMETRY_ALWAYS_EXPECT(x2 != x0);

      Vec3 a = x2 - x1;
      Vec3 b = x0 - x1;

      const Real aLen = a.length();
      const Real bLen = b.length();

      EBGEOMETRY_ALWAYS_EXPECT(aLen > EBGeometry::Limits::eps());
      EBGEOMETRY_ALWAYS_EXPECT(bLen > EBGeometry::Limits::eps());

      // Add this face -- if the spanned triangle is degenerate then we skip this face.
      if (aLen > EBGeometry::Limits::eps() && bLen > EBGeometry::Limits::eps()) {
        a = a / aLen;
        b = b / bLen;

        const Vec3& faceNormal = m_faceList[faceIndex].getNormal();
        const Real  dotProduct = dot(a, b);
        const Real  clampedDot = EBGeometry::min(Real(1.0), EBGeometry::max(Real(-1.0), dotProduct));
        const Real  alpha      = static_cast<Real>(acos(clampedDot));

        EBGEOMETRY_ALWAYS_EXPECT(faceNormal.length() > 0.0);

        m_normal += alpha * faceNormal;
      }

      // Jump to the pair polygon.
      outgoingEdge = m_edgeList[outgoingEdge].getPairEdge();
      EBGEOMETRY_ALWAYS_EXPECT(outgoingEdge >= 0 && outgoingEdge < m_edgeList.length());

      // Fetch the edge in the next polygon which has this vertex
      // as the starting vertex.
      outgoingEdge = m_edgeList[outgoingEdge].getNextEdge();
      EBGEOMETRY_ALWAYS_EXPECT(outgoingEdge >= 0 && outgoingEdge < m_edgeList.length());
    }

    // Ensure accumulated normal has non-zero length before normalizing
    // (prevents division by zero when angle-weighted normals cancel out)
    EBGEOMETRY_ALWAYS_EXPECT(m_normal.length() > EBGeometry::Limits::eps());

    this->normalizeNormalVector();
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  Vec3&
  Vertex<MetaData>::getPosition() noexcept
  {
    return m_position;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  const Vec3&
  Vertex<MetaData>::getPosition() const noexcept
  {
    return m_position;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  Vec3&
  Vertex<MetaData>::getNormal() noexcept
  {
    return m_normal;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  const Vec3&
  Vertex<MetaData>::getNormal() const noexcept
  {
    return m_normal;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  int
  Vertex<MetaData>::getOutgoingEdge() const noexcept
  {
    return m_outgoingEdge;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  Real
  Vertex<MetaData>::signedDistance(const Vec3& a_x0) const noexcept
  {
    const Vec3 delta = a_x0 - m_position;
    const Real dist  = delta.length();
    const Real proj  = dot(m_normal, delta);
    const int  sign  = EBGeometry::sgn(proj);

    return (dist != 0.0) ? Real(sign) * dist : 0.0;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  Real
  Vertex<MetaData>::unsignedDistance2(const Vec3& a_x0) const noexcept
  {

    return (a_x0 - m_position).length2();
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  MetaData&
  Vertex<MetaData>::getMetaData() noexcept
  {
    return m_metaData;
  }

  template <class MetaData>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  const MetaData&
  Vertex<MetaData>::getMetaData() const noexcept
  {
    return m_metaData;
  }
} // namespace EBGeometry::DCEL

#endif
