/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_DCEL_VertexImplem.hpp
  @brief  Implementation of EBGeometry_DCEL_Vertex.hpp
  @author Robert Marskar
*/

#ifndef EBGeometry_DCEL_VertexImplem
#define EBGeometry_DCEL_VertexImplem

// Std includes
#include <cmath>

// Our includes
#include "EBGeometry_DCEL_Vertex.hpp"
#include "EBGeometry_Macros.hpp"

namespace EBGeometry {
  namespace DCEL {

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    Vertex<MetaData>::Vertex() noexcept
    {
      m_position     = Vec3::zero();
      m_normal       = Vec3::zero();
      m_outgoingEdge = -1;
      m_vertexList   = nullptr;
      m_edgeList     = nullptr;
      m_faceList     = nullptr;
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    Vertex<MetaData>::Vertex(const Vec3& a_position) noexcept :
      Vertex<MetaData>()
    {
      m_position = a_position;
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    Vertex<MetaData>::Vertex(const Vec3& a_position, const Vec3& a_normal) noexcept :
      Vertex<MetaData>()
    {
      m_position = a_position;
      m_normal   = a_normal;
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    Vertex<MetaData>::Vertex(const Vec3& a_position, const Vec3& a_normal, const int a_edge) noexcept :
      Vertex<MetaData>()
    {
      this->define(a_position, a_normal, a_edge);
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    Vertex<MetaData>::Vertex(const Vertex& a_otherVertex) noexcept
    {
      m_position     = a_otherVertex.m_position;
      m_normal       = a_otherVertex.m_normal;
      m_outgoingEdge = a_otherVertex.m_outgoingEdge;
      m_vertexList   = a_otherVertex.m_vertexList;
      m_edgeList     = a_otherVertex.m_edgeList;
      m_faceList     = a_otherVertex.m_faceList;
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    Vertex<MetaData>::~Vertex() noexcept
    {}

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    void
    Vertex<MetaData>::define(const Vec3& a_position, const Vec3& a_normal, const int a_edge) noexcept
    {
      m_position     = a_position;
      m_normal       = a_normal;
      m_outgoingEdge = a_edge;
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    void
    Vertex<MetaData>::setPosition(const Vec3& a_position) noexcept
    {
      m_position = a_position;
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    void
    Vertex<MetaData>::setNormal(const Vec3& a_normal) noexcept
    {
      m_normal = a_normal;
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    void
    Vertex<MetaData>::setEdge(const int a_edge) noexcept
    {
      m_outgoingEdge = a_edge;
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    void
    Vertex<MetaData>::setMetaData(const MetaData& a_metaData) noexcept
    {
      m_metaData = a_metaData;
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    void
    Vertex<MetaData>::setVertexList(const Vertex<MetaData>* const a_vertexList) noexcept
    {
      m_vertexList = a_vertexList;
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    void
    Vertex<MetaData>::setEdgeList(const Edge<MetaData>* const a_edgeList) noexcept
    {
      m_edgeList = a_edgeList;
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    void
    Vertex<MetaData>::setFaceList(const Face<MetaData>* const a_faceList) noexcept
    {
      m_faceList = a_faceList;
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    const Vertex<MetaData>*
    Vertex<MetaData>::getVertexList() const noexcept
    {
      return (m_vertexList);
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    const Edge<MetaData>*
    Vertex<MetaData>::getEdgeList() const noexcept
    {
      return (m_edgeList);
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    const Face<MetaData>*
    Vertex<MetaData>::getFaceList() const noexcept
    {
      return (m_faceList);
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    void
    Vertex<MetaData>::normalizeNormalVector() noexcept
    {
      EBGEOMETRY_EXPECT(m_normal.length() > 0.0);

      m_normal = m_normal / m_normal.length();
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    void
    Vertex<MetaData>::computeVertexNormalAverage() noexcept
    {
      // This routine computes the normal vector using a weighted sum of all faces
      // that share this vertex.
      EBGEOMETRY_EXPECT(m_outgoingEdge >= 0);
      EBGEOMETRY_EXPECT(m_vertexList != nullptr);
      EBGEOMETRY_EXPECT(m_edgeList != nullptr);
      EBGEOMETRY_EXPECT(m_faceList != nullptr);

      m_normal = Vec3::zero();

      int curEdge = -1;
      int curFace = -1;

      while (curEdge != m_outgoingEdge) {
        curEdge = (curEdge < 0) ? m_outgoingEdge : curEdge;
        curFace = m_edgeList[curEdge].getFace();

        m_normal += m_faceList[curFace].getNormal();

        // Jump to the pair edge and advance so we get the outgoing edge (from this vertex) on
        // the next polygon.
        curEdge = m_edgeList[curEdge].getPairEdge();
        EBGEOMETRY_EXPECT(curEdge >= 0);

        curEdge = m_edgeList[curEdge].getNextEdge();
        EBGEOMETRY_EXPECT(curEdge >= 0);
      }

      this->normalizeNormalVector();
    }

    template <class MetaData>
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
      EBGEOMETRY_EXPECT(m_outgoingEdge >= 0);
      EBGEOMETRY_EXPECT(m_vertexList != nullptr);
      EBGEOMETRY_EXPECT(m_edgeList != nullptr);
      EBGEOMETRY_EXPECT(m_faceList != nullptr);

      int outgoingEdge = -1;
      int incomingEdge = -1;
      int faceIndex    = -1;

      while (outgoingEdge != m_outgoingEdge) {

        // Get the incoming and outgoing edges out of the origin vertex.
        outgoingEdge = (outgoingEdge < 0) ? m_outgoingEdge : outgoingEdge;
        incomingEdge = m_edgeList[outgoingEdge].getPreviousEdge();
        faceIndex    = m_edgeList[outgoingEdge].getFace();

        EBGEOMETRY_EXPECT(outgoingEdge >= 0);
        EBGEOMETRY_EXPECT(incomingEdge >= 0);
        EBGEOMETRY_EXPECT(outgoingEdge != incomingEdge);
        EBGEOMETRY_EXPECT(faceIndex >= 0);

        // Vertices are named v0,v1,v2:
        // v0 = Origin vertex of incoming edge
        // v1 = this vertex
        // v2 = End vertex of outgoing edge.
        const int v0 = m_edgeList[incomingEdge].getVertex();
        const int v2 = m_edgeList[outgoingEdge].getOtherVertex();

        EBGEOMETRY_EXPECT(v0 != v2);
        EBGEOMETRY_EXPECT(v0 >= 0);
        EBGEOMETRY_EXPECT(v2 >= 0);

        const Vec3& x0 = m_vertexList[v0].getPosition();
        const Vec3& x1 = m_position;
        const Vec3& x2 = m_vertexList[v2].getPosition();

        EBGEOMETRY_EXPECT(x0 != x1);
        EBGEOMETRY_EXPECT(x1 != x2);
        EBGEOMETRY_EXPECT(x2 != x0);

        Vec3 a = x2 - x1;
        Vec3 b = x0 - x1;

        a = a / a.length();
        b = b / b.length();

        const Vec3& faceNormal = m_faceList[faceIndex].getNormal();
        const Real  alpha      = acos(dot(a, b));

        EBGEOMETRY_EXPECT(faceNormal.length() > 0.0);

        m_normal += alpha * faceNormal;

        // Jump to the pair polygon.
        outgoingEdge = m_edgeList[outgoingEdge].getPairEdge();
        EBGEOMETRY_EXPECT(outgoingEdge >= 0);

        // Fetch the edge in the next polygon which has this vertex
        // as the starting vertex.
        outgoingEdge = m_edgeList[outgoingEdge].getNextEdge();
        EBGEOMETRY_EXPECT(outgoingEdge >= 0);
      }

      this->normalizeNormalVector();
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    int
    Vertex<MetaData>::getEdge() const noexcept
    {
      return (m_outgoingEdge);
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    Vec3&
    Vertex<MetaData>::getPosition() noexcept
    {
      return (m_position);
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    const Vec3&
    Vertex<MetaData>::getPosition() const noexcept
    {
      return (m_position);
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    Vec3&
    Vertex<MetaData>::getNormal() noexcept
    {
      return (m_normal);
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    const Vec3&
    Vertex<MetaData>::getNormal() const noexcept
    {
      return (m_normal);
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    int
    Vertex<MetaData>::getOutgoingEdge() const noexcept
    {
      return m_outgoingEdge;
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    Real
    Vertex<MetaData>::signedDistance(const Vec3& a_x0) const noexcept
    {
      const Vec3 delta = a_x0 - m_position;
      const Real dist  = delta.length();
      const Real dot   = m_normal.dot(delta);
      const int  sign  = (dot > 0.0) ? 1 : -1;

      return dist * sign;
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    Real
    Vertex<MetaData>::unsignedDistance2(const Vec3& a_x0) const noexcept
    {
      const Vec3 delta = a_x0 - m_position;

      return delta.dot(delta);
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    MetaData&
    Vertex<MetaData>::getMetaData() noexcept
    {
      return (m_metaData);
    }

    template <class MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    const MetaData&
    Vertex<MetaData>::getMetaData() const noexcept
    {
      return (m_metaData);
    }
  } // namespace DCEL
} // namespace EBGeometry

#endif
