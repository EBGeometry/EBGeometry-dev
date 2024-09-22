/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_DCEL_Vertex.hpp
  @brief  Declaration of a vertex class for use in DCEL descriptions of polygon
  tesselations.
  @author Robert Marskar
*/

#ifndef EBGeometry_DCEL_VertexImplem
#define EBGeometry_DCEL_VertexImplem

// Our includes
#include "EBGeometry_DCEL_Vertex.hpp"

namespace EBGeometry {
  namespace DCEL {

    template <class Meta>
    inline Vertex<Meta>::Vertex() noexcept
    {
      m_position     = Vec3::zero();
      m_normal       = Vec3::zero();
      m_outgoingEdge = -1;
    }

    template <class Meta>
    inline Vertex<Meta>::Vertex(const Vec3& a_position) noexcept
    {
      m_position     = a_position;
      m_normal       = Vec3::zero();
      m_outgoingEdge = -1;
    }

    template <class Meta>
    inline Vertex<Meta>::Vertex(const Vec3& a_position, const Vec3& a_normal) noexcept
    {
      m_position     = a_position;
      m_normal       = a_normal;
      m_outgoingEdge = -1;
    }

    template <class Meta>
    inline Vertex<Meta>::Vertex(const Vec3& a_position, const Vec3& a_normal, const int& a_edge) noexcept
    {
      m_position     = a_position;
      m_normal       = a_normal;
      m_outgoingEdge = a_edge;
    }

    template <class Meta>
    inline Vertex<Meta>::Vertex(const Vertex& a_otherVertex) noexcept
    {
      m_position     = a_otherVertex.m_position;
      m_normal       = a_otherVertex.m_normal;
      m_outgoingEdge = a_otherVertex.m_outgoingEdge;
    }

    template <class Meta>
    inline Vertex<Meta>::~Vertex() noexcept
    {}

    template <class Meta>
    inline void
    Vertex<Meta>::define(const Vec3& a_position, const Vec3& a_normal, const int& a_edge) noexcept
    {
      m_position     = a_position;
      m_normal       = a_normal;
      m_outgoingEdge = a_edge;
    }

    template <class Meta>
    inline void
    Vertex<Meta>::setPosition(const Vec3& a_position) noexcept
    {
      m_position = a_position;
    }

    template <class Meta>
    inline void
    Vertex<Meta>::setNormal(const Vec3& a_normal) noexcept
    {
      m_normal = a_normal;
    }

    template <class Meta>
    inline void
    Vertex<Meta>::setEdge(const int& a_edge) noexcept
    {
      m_outgoingEdge = a_edge;
    }

    template <class Meta>
    inline void
    Vertex<Meta>::normalizeNormalVector() noexcept
    {
      m_normal = m_normal / m_normal.length();
    }

    template <class Meta>
    inline void
    Vertex<Meta>::computeVertexNormalAverage() noexcept
    {
#warning "Vertex<Meta>::computeVertexNormalAverage is not implemented"
    }

    template <class Meta>
    inline void
    Vertex<Meta>::computeVertexNormalAngleWeighted() noexcept
    {
#warning "Vertex<Meta>::computeVertexNormalAngleWeighted is not implemented"
    }

    template <class Meta>
    inline void
    Vertex<Meta>::flip() noexcept
    {
      m_normal = -m_normal;
    }

    template <class Meta>
    inline Vec3&
    Vertex<Meta>::getPosition() noexcept
    {
      return (m_position);
    }

    template <class Meta>
    inline const Vec3&
    Vertex<Meta>::getPosition() const noexcept
    {
      return (m_position);
    }

    template <class Meta>
    inline Vec3&
    Vertex<Meta>::getNormal() noexcept
    {
      return (m_normal);
    }

    template <class Meta>
    inline const Vec3&
    Vertex<Meta>::getNormal() const noexcept
    {
      return (m_normal);
    }

    template <class Meta>
    inline int&
    Vertex<Meta>::getOutgoingEdge() noexcept
    {
      return (m_outgoingEdge);
    }

    template <class Meta>
    inline const int&
    Vertex<Meta>::getOutgoingEdge() const noexcept
    {
      return (m_outgoingEdge);
    }

    template <class Meta>
    inline Real
    Vertex<Meta>::signedDistance(const Vec3& a_x0) const noexcept
    {
      const Vec3 delta = a_x0 - m_position;
      const Real dist  = delta.length();
      const Real dot   = m_normal.dot(delta);
      const int  sign  = (dot > 0.0) ? 1 : -1;

      return dist * sign;
    }

    template <class Meta>
    inline Real
    Vertex<Meta>::unsignedDistance2(const Vec3& a_x0) const noexcept
    {
      const Vec3 delta = a_x0 - m_position;

      return delta.dot(delta);
    }

    template <class Meta>
    inline Meta&
    Vertex<Meta>::getMetaData() noexcept
    {
      return (m_metaData);
    }

    template <class Meta>
    inline const Meta&
    Vertex<Meta>::getMetaData() const noexcept
    {
      return (m_metaData);
    }
  } // namespace DCEL
} // namespace EBGeometry

#include "EBGeometry_DCEL_VertexImplem.hpp"

#endif
