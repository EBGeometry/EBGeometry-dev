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
      m_vertex   = -1;
      m_pairEdge = -1;
      m_nextEdge = -1;
      m_face     = -1;
      m_normal   = Vec3::zero();
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
    inline Edge<Meta>::Edge(const int a_vertex) : Edge() noexcept
    {
      m_vertex = a_vertex;
    }

    template <class Meta>
    inline Edge<Meta>::Edge(const int a_vertex, const int a_pairEdge, const int a_nextEdge, const int a_face) noexcept
    {}

    template <class Meta>
    inline Edge<Meta>::~Edge() noexcept
    {}

    template <class Meta>
    inline void
    Edge<Meta>::define(const int a_vertex, const int a_pairEdge, const int a_nextEdge, const int a_face) noexcept
    {}

    template <class Meta>
    inline void
    Edge<Meta>::setVertex(const int a_vertex) noexcept
    {}

    template <class Meta>
    inline void
    Edge<Meta>::setPairEdge(const int a_pairEdge) noexcept
    {}

    template <class Meta>
    inline void
    Edge<Meta>::setNextEdge(const int a_nextEdge) noexcept
    {}

    template <class Meta>
    inline void
    Edge<Meta>::setFace(const int a_face) noexcept
    {}

    template <class Meta>
    inline void
    Edge<Meta>::flip() noexcept
    {}

    template <class Meta>
    inline int
    Edge<Meta>::getVertex() const noexcept
    {}

    template <class Meta>
    inline int
    Edge<Meta>::getPairEdge() const noexcept
    {}

    template <class Meta>
    inline int
    Edge<Meta>::getNextEdge() const noexcept
    {}

    template <class Meta>
    inline Vec3
    Edge<Meta>::computeNormal() const noexcept
    {}

    template <class Meta>
    inline const Vec3&
    Edge<Meta>::getNormal() const noexcept
    {}

    template <class Meta>
    inline int
    Edge<Meta>::getFace() const noexcept
    {}

    template <class Meta>
    inline Meta&
    Edge<Meta>::getMetaData() noexcept
    {}

    template <class Meta>
    inline const Meta&
    Edge<Meta>::getMetaData() const noexcept
    {}

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
