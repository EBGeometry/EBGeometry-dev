/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_DCEL_EdgeImplem.hpp
  @brief  Implementation of EBGeometry_DCEL_Edge.hpp
  @author Robert Marskar
*/

#ifndef EBGeometry_DCEL_EdgeImplem
#define EBGeometry_DCEL_EdgeImplem

// Our includes
#include "EBGeometry_DCEL_Edge.hpp"
#include "EBGeometry_Macros.hpp"

namespace EBGeometry {
  namespace DCEL {

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE
    Edge<Meta>::Edge() noexcept
    {
      this->define(-1, -1, -1, -1, -1);

      m_vertexList = nullptr;
      m_edgeList   = nullptr;
      m_faceList   = nullptr;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE
    Edge<Meta>::Edge(const Edge& a_otherEdge) noexcept
    {
      m_vertex     = a_otherEdge.m_vertex;
      m_pairEdge   = a_otherEdge.m_previousEdge;
      m_pairEdge   = a_otherEdge.m_pairEdge;
      m_nextEdge   = a_otherEdge.m_nextEdge;
      m_face       = a_otherEdge.m_face;
      m_normal     = a_otherEdge.m_normal;
      m_vertexList = a_otherEdge.m_vertexList;
      m_edgeList   = a_otherEdge.m_edgeList;
      m_faceList   = a_otherEdge.m_faceList;
      m_metaData   = a_otherEdge.m_metaData;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE
    Edge<Meta>::Edge(const int a_vertex) noexcept
      : Edge()
    {
      m_vertex = a_vertex;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE
    Edge<Meta>::Edge(const int a_vertex,
                     const int a_previousEdge,
                     const int a_pairEdge,
                     const int a_nextEdge,
                     const int a_face) noexcept
      : Edge()
    {
      this->define(a_vertex, a_previousEdge, a_pairEdge, a_nextEdge, a_face);
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE Edge<Meta>::~Edge() noexcept
    {}

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Edge<Meta>::define(const int a_vertex,
                       const int a_previousEdge,
                       const int a_pairEdge,
                       const int a_nextEdge,
                       const int a_face) noexcept
    {
      this->setVertex(a_vertex);
      this->setPreviousEdge(a_previousEdge);
      this->setPairEdge(a_pairEdge);
      this->setNextEdge(a_nextEdge);
      this->setFace(a_face);
      this->setNormal(Vec3::zero());
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Edge<Meta>::setVertex(const int a_vertex) noexcept
    {
      m_vertex = a_vertex;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Edge<Meta>::setPreviousEdge(const int a_previousEdge) noexcept
    {
      m_previousEdge = a_previousEdge;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Edge<Meta>::setPairEdge(const int a_pairEdge) noexcept
    {
      m_pairEdge = a_pairEdge;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Edge<Meta>::setNextEdge(const int a_nextEdge) noexcept
    {
      m_nextEdge = a_nextEdge;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Edge<Meta>::setFace(const int a_face) noexcept
    {
      m_face = a_face;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Edge<Meta>::setVertexList(const Vertex<Meta>* const a_vertexList) noexcept
    {
      m_vertexList = a_vertexList;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Edge<Meta>::setEdgeList(const Edge<Meta>* const a_edgeList) noexcept
    {
      m_edgeList = a_edgeList;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Edge<Meta>::setFaceList(const Face<Meta>* const a_faceList) noexcept
    {
      m_faceList = a_faceList;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Edge<Meta>::setNormal(const Vec3& a_normal) noexcept
    {
      m_normal = a_normal;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Edge<Meta>::computeNormal() noexcept
    {
      EBGEOMETRY_EXPECT(m_edgeList != nullptr);
      EBGEOMETRY_EXPECT(m_faceList != nullptr);
      EBGEOMETRY_EXPECT(m_face >= 0);
      EBGEOMETRY_EXPECT(m_pairEdge >= 0);
      EBGEOMETRY_EXPECT(m_faceList[m_pairEdge].getFace() >= 0);

      m_normal = Vec3::zero();
      m_normal += m_faceList[m_face].getNormal();
      m_normal += m_faceList[m_edgeList[m_pairEdge].getFace()].getNormal();

      this->normalizeNormalVector();
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE int
    Edge<Meta>::getVertex() const noexcept
    {
      return m_vertex;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE int
    Edge<Meta>::getOtherVertex() const noexcept
    {
      EBGEOMETRY_EXPECT(m_edgeList != nullptr);
      EBGEOMETRY_EXPECT(m_nextEdge >= 0);

      return m_edgeList[m_nextEdge].getVertex();
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE int
    Edge<Meta>::getPreviousEdge() const noexcept
    {
      return m_previousEdge;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE int
    Edge<Meta>::getPairEdge() const noexcept
    {
      return m_pairEdge;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE int
    Edge<Meta>::getNextEdge() const noexcept
    {
      return m_nextEdge;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE int
    Edge<Meta>::getFace() const noexcept
    {
      return m_face;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE const Vec3&
    Edge<Meta>::getNormal() const noexcept
    {
      return (m_normal);
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE Meta&
    Edge<Meta>::getMetaData() noexcept
    {
      return (m_metaData);
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE const Meta&
    Edge<Meta>::getMetaData() const noexcept
    {
      return (m_metaData);
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE Vec3
    Edge<Meta>::getX2X1() const noexcept
    {
      EBGEOMETRY_EXPECT(m_vertexList != nullptr);
      EBGEOMETRY_EXPECT(this->getVertex() >= 0);
      EBGEOMETRY_EXPECT(this->getOtherVertex() >= 0);

      const auto& x1 = m_vertexList[this->getVertex()].getPosition();
      const auto& x2 = m_vertexList[this->getOtherVertex()].getPosition();

      return x2 - x1;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE Real
    Edge<Meta>::projectPointToEdge(const Vec3& a_x0) const noexcept
    {
      EBGEOMETRY_EXPECT(m_vertexList != nullptr);
      EBGEOMETRY_EXPECT(m_vertex >= 0);

      const auto p    = a_x0 - m_vertexList[m_vertex].getPosition();
      const auto x2x1 = this->getX2X1();

      return p.dot(x2x1) / (x2x1.dot(x2x1));
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE Real
    Edge<Meta>::signedDistance(const Vec3& a_x0) const noexcept
    {
      EBGEOMETRY_EXPECT(m_vertexList != nullptr);
      EBGEOMETRY_EXPECT(this->getVertex() >= 0);
      EBGEOMETRY_EXPECT(this->getOtherVertex() >= 0);

      // Project point to edge
      const Real t = this->projectPointToEdge(a_x0);

      Real retval = 0.0;

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

        retval = sgn * delta.length();
      }

      return retval;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE Real
    Edge<Meta>::unsignedDistance2(const Vec3& a_x0) const noexcept
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
  } // namespace DCEL
} // namespace EBGeometry

#endif
