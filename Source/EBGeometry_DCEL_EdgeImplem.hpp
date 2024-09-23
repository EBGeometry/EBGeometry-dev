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

namespace EBGeometry {
  namespace DCEL {

    template <class Meta>
    inline Edge<Meta>::Edge() noexcept
    {
      this->define(-1, -1, -1, -1);
    }

    template <class Meta>
    inline Edge<Meta>::Edge(const Edge& a_otherEdge) noexcept
    {
      m_vertex   = a_otherEdge.m_vertex;
      m_pairEdge = a_otherEdge.m_pairEdge;
      m_nextEdge = a_otherEdge.m_nextEdge;
      m_face     = a_otherEdge.m_face;
      m_normal   = a_otherEdge.m_normal;
    }

    template <class Meta>
    inline Edge<Meta>::Edge(const int a_vertex) noexcept : Edge()
    {
      m_vertex = a_vertex;
    }

    template <class Meta>
    inline Edge<Meta>::Edge(const int a_vertex, const int a_pairEdge, const int a_nextEdge, const int a_face) noexcept
      : Edge()
    {
      this->define(a_vertex, a_pairEdge, a_nextEdge, a_face);
    }

    template <class Meta>
    inline Edge<Meta>::~Edge() noexcept
    {}

    template <class Meta>
    inline void
    Edge<Meta>::define(const int a_vertex, const int a_pairEdge, const int a_nextEdge, const int a_face) noexcept
    {
      this->setVertex(a_vertex);
      this->setPairEdge(a_pairEdge);
      this->setNextEdge(a_nextEdge);
      this->setFace(a_face);
      this->setNormal(Vec3::one());
    }

    template <class Meta>
    inline void
    Edge<Meta>::setVertex(const int a_vertex) noexcept
    {
      m_vertex = a_vertex;
    }

    template <class Meta>
    inline void
    Edge<Meta>::setPairEdge(const int a_pairEdge) noexcept
    {
      m_pairEdge = a_pairEdge;
    }

    template <class Meta>
    inline void
    Edge<Meta>::setNextEdge(const int a_nextEdge) noexcept
    {
      m_nextEdge = a_nextEdge;
    }

    template <class Meta>
    inline void
    Edge<Meta>::setFace(const int a_face) noexcept
    {
      m_face = a_face;
    }

    template <class Meta>
    inline void
    Edge<Meta>::setNormal(const Vec3& a_normal) noexcept
    {
      m_normal = a_normal;
    }

    template <class Meta>
    inline void
    Edge<Meta>::flip() noexcept
    {
      m_normal = -m_normal;
    }

    template <class Meta>
    inline int
    Edge<Meta>::getVertex() const noexcept
    {
      return m_vertex;
    }

    template <class Meta>
    inline int
    Edge<Meta>::getPairEdge() const noexcept
    {
      return m_pairEdge;
    }

    template <class Meta>
    inline int
    Edge<Meta>::getNextEdge() const noexcept
    {
      return m_nextEdge;
    }

    template <class Meta>
    inline const Vec3&
    Edge<Meta>::getNormal() const noexcept
    {
      return (m_normal);
    }

    template <class Meta>
    inline int
    Edge<Meta>::getFace() const noexcept
    {
      return m_face;
    }

    template <class Meta>
    inline Meta&
    Edge<Meta>::getMetaData() noexcept
    {
      return (m_metaData);
    }

    template <class Meta>
    inline const Meta&
    Edge<Meta>::getMetaData() const noexcept
    {
      return (m_metaData);
    }

    template <class Meta>
    inline Real
    Edge<Meta>::projectPointToEdge(const Vec3& a_x0) const noexcept
    {
#warning "Edge<Meta>::projectPointToEdge -- not implemented"
    }

    template <class Meta>
    inline Vec3
    Edge<Meta>::getX2X1() const noexcept
    {
#warning "Edge<Meta>::getX2X1 -- x2x1 should be a member. This function is not yet implemented"
    }

    template <class Meta>
    inline Real
    Edge<Meta>::signedDistance(const Vec3& a_x0) const noexcept
    {}

    template <class Meta>
    inline Real
    Edge<Meta>::unsignedDistance2(const Vec3& a_x0) const noexcept
    {}
  } // namespace DCEL
} // namespace EBGeometry

#include "EBGeometry_DCEL_VertexImplem.hpp"

#endif
