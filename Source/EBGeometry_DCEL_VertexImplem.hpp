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

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE
    Vertex<Meta>::Vertex() noexcept
    {
      m_position     = Vec3::zero();
      m_normal       = Vec3::zero();
      m_outgoingEdge = -1;
      m_edgeList     = nullptr;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE
    Vertex<Meta>::Vertex(const Vec3& a_position) noexcept
      : Vertex<Meta>()
    {
      m_position = a_position;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE
    Vertex<Meta>::Vertex(const Vec3& a_position, const Vec3& a_normal) noexcept
      : Vertex<Meta>()
    {
      m_position = a_position;
      m_normal   = a_normal;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE
    Vertex<Meta>::Vertex(const Vec3& a_position, const Vec3& a_normal, const int a_edge) noexcept
      : Vertex<Meta>()
    {
      this->define(a_position, a_normal, a_edge);
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE
    Vertex<Meta>::Vertex(const Vertex& a_otherVertex) noexcept
    {
      m_position     = a_otherVertex.m_position;
      m_normal       = a_otherVertex.m_normal;
      m_outgoingEdge = a_otherVertex.m_outgoingEdge;
      m_edgeList     = a_otherVertex.m_edgeList;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE Vertex<Meta>::~Vertex() noexcept
    {}

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Vertex<Meta>::define(const Vec3& a_position, const Vec3& a_normal, const int a_edge) noexcept
    {
      m_position     = a_position;
      m_normal       = a_normal;
      m_outgoingEdge = a_edge;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Vertex<Meta>::setPosition(const Vec3& a_position) noexcept
    {
      m_position = a_position;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Vertex<Meta>::setNormal(const Vec3& a_normal) noexcept
    {
      m_normal = a_normal;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Vertex<Meta>::setEdge(const int a_edge) noexcept
    {
      m_outgoingEdge = a_edge;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Vertex<Meta>::setEdgeList(const Edge<Meta>* const a_edgeList) noexcept
    {
      m_edgeList = a_edgeList;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Vertex<Meta>::normalizeNormalVector() noexcept
    {
      EBGEOMETRY_EXPECT(m_normal.length() > EBGeometry::Limits::min());

      m_normal = m_normal / m_normal.length();
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Vertex<Meta>::computeVertexNormalAverage() noexcept
    {
      // This routine computes the normal vector using a weighted sum of all faces
      // that share this vertex.
      EBGEOMETRY_EXPECT(m_outgoingEdge >= 0);
      EBGEOMETRY_EXPECT(m_edgeList != nullptr);

#if 1
#warning \
  "EBGeometry_DCEL_VertexImplem::computeVertexNormalAverage is not implemented (maybe move this to external functionality?"
#else
      const Edge<Meta>* outgoingEdge = nullptr;

      m_normal = Vec3::zero();

      while (outgoingEdge != m_outgoingEdge) {
        outgoingEdge = (outgoingEdge == nullptr) ? m_outgoingEdge : outgoingEdge;

        const Vec3& faceNormal = (outgoingEdge->getFace())->getNormal();

        m_normal += faceNormal;

        // Jump to the pair edge and advance so we get the outgoing edge (from this vertex) on
        // the next polygon.
        outgoingEdge = outgoingEdge->getPairEdge();
        EBGEOMETRY_EXPECT(outgoingEdge != nullptr);

        outgoingEdge = outgoingEdge->getNextEdge();
        EBGEOMETRY_EXPECT(outgoingEdge != nullptr);
      }
#endif

      this->normalizeNormalVector();
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Vertex<Meta>::computeVertexNormalAngleWeighted() noexcept
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
      EBGEOMETRY_EXPECT(m_edgeList != nullptr);

#if 1
#warning \
  "EBGeometry_DCEL_VertexImplem::computeVertexNormalAngleWeighted is not implemented (maybe move this to external functionality?      
#else

      const Edge<Meta>* outgoingEdge = nullptr;
      const Edge<Meta>* incomingEdge = nullptr;

      while (outgoingEdge != m_outgoingEdge) {

        // Get the incoming and outgoing edges out of the origin vertex.
        outgoingEdge = (outgoingEdge == nullptr) ? m_outgoingEdge : outgoingEdge;
        incomingEdge = outgoingEdge->getPreviousEdge();

        EBGEOMETRY_EXPECT(outgoingEdge != nullptr);
        EBGEOMETRY_EXPECT(incomingEdge != nullptr);

        // Vertices are named v0,v1,v2:
        // v0 = Origin vertex of incoming edge
        // v1 = this vertex
        // v2 = End vertex of outgoing edge.
        const Vertex<Meta>* const v0 = outgoingEdge->getVertex();
        const Vertex<Meta>* const v1 = this;
        const Vertex<Meta>* const v2 = outgoingEdge->getOtherVertex();

        EBGEOMETRY_EXPECT(v0 != v1);
        EBGEOMETRY_EXPECT(v1 != v2);
        EBGEOMETRY_EXPECT(v2 != v0);

        const Vec3& x0 = v0->getPosition();
        const Vec3& x1 = v1->getPosition();
        const Vec3& x2 = v2->getPosition();

        EBGEOMETRY_EXPECT(x0 != x1);
        EBGEOMETRY_EXPECT(x1 != x2);
        EBGEOMETRY_EXPECT(x2 != x0);

        Vec3 a = x2 - x1;
        Vec3 b = x0 - x1;

        a = a / a.length();
        b = b / b.length();

        const Vec3& faceNormal = (outgoingEdge->getFace())->getNormal();
        const Real  alpha      = acos(dot(a, b));

        m_normal += alpha * faceNormal;

        // Jump to the pair polygon.
        outgoingEdge = outgoingEdge->getPairEdge();
        EBGEOMETRY_EXPECT(outgoingEdge != nullptr);

        // Fetch the edge in the next polygon which has this vertex
        // as the starting vertex.
        outgoingEdge = outgoingEdge->getNextEdge();
        EBGEOMETRY_EXPECT(outgoingEdge != nullptr);
      }
#endif

      this->normalizeNormalVector();
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE Vec3&
    Vertex<Meta>::getPosition() noexcept
    {
      return (m_position);
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE const Vec3&
    Vertex<Meta>::getPosition() const noexcept
    {
      return (m_position);
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE Vec3&
    Vertex<Meta>::getNormal() noexcept
    {
      return (m_normal);
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE const Vec3&
    Vertex<Meta>::getNormal() const noexcept
    {
      return (m_normal);
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE int
    Vertex<Meta>::getOutgoingEdge() const noexcept
    {
      return m_outgoingEdge;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE Real
    Vertex<Meta>::signedDistance(const Vec3& a_x0) const noexcept
    {
      const Vec3 delta = a_x0 - m_position;
      const Real dist  = delta.length();
      const Real dot   = m_normal.dot(delta);
      const int  sign  = (dot > 0.0) ? 1 : -1;

      return dist * sign;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE Real
    Vertex<Meta>::unsignedDistance2(const Vec3& a_x0) const noexcept
    {
      const Vec3 delta = a_x0 - m_position;

      return delta.dot(delta);
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE Meta&
    Vertex<Meta>::getMetaData() noexcept
    {
      return (m_metaData);
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE const Meta&
    Vertex<Meta>::getMetaData() const noexcept
    {
      return (m_metaData);
    }
  } // namespace DCEL
} // namespace EBGeometry

#endif
