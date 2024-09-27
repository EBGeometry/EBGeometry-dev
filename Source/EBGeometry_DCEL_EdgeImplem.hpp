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
      this->define(nullptr, nullptr, nullptr, nullptr);
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
    inline Edge<Meta>::Edge(const VertexPointer a_vertex) noexcept : Edge()
    {
      m_vertex = a_vertex;
    }

    template <class Meta>
    inline Edge<Meta>::Edge(const VertexPointer a_vertex,
                            const EdgePointer   a_pairEdge,
                            const EdgePointer   a_nextEdge,
                            const FacePointer   a_face) noexcept
      : Edge()
    {
      this->define(a_vertex, a_pairEdge, a_nextEdge, a_face);
    }

    template <class Meta>
    inline Edge<Meta>::~Edge() noexcept
    {}

    template <class Meta>
    inline void
    Edge<Meta>::define(const VertexPointer a_vertex,
                       const EdgePointer   a_pairEdge,
                       const EdgePointer   a_nextEdge,
                       const FacePointer   a_face) noexcept
    {
      this->setVertex(a_vertex);
      this->setPairEdge(a_pairEdge);
      this->setNextEdge(a_nextEdge);
      this->setFace(a_face);
      this->setNormal(Vec3::zero());
    }

    template <class Meta>
    inline void
    Edge<Meta>::setVertex(const VertexPointer a_vertex) noexcept
    {
      m_vertex = a_vertex;
    }

    template <class Meta>
    inline void
    Edge<Meta>::setPairEdge(const EdgePointer a_pairEdge) noexcept
    {
      m_pairEdge = a_pairEdge;
    }

    template <class Meta>
    inline void
    Edge<Meta>::setNextEdge(const EdgePointer a_nextEdge) noexcept
    {
      m_nextEdge = a_nextEdge;
    }

    template <class Meta>
    inline void
    Edge<Meta>::setFace(const FacePointer a_face) noexcept
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
    Edge<Meta>::computeNormal() noexcept
    {
      m_normal = Vec3::zero();

      if (m_face != nullptr) {
        m_normal += m_face->getNormal();
      }
      if (m_pairEdge != nullptr) {
        m_normal += (m_pairEdge->getFace())->getNormal();
      }

      this->normalizeNormalVector();
    }

    template <class Meta>
    inline Edge<Meta>::VertexPointer
    Edge<Meta>::getVertex() const noexcept
    {
      return m_vertex;
    }

    template <class Meta>
    inline Edge<Meta>::VertexPointer
    Edge<Meta>::getOtherVertex() const noexcept
    {
      return (m_nextEdge->getVertex());
    }

    template <class Meta>
    inline Edge<Meta>::EdgePointer
    Edge<Meta>::getPairEdge() const noexcept
    {
      return m_pairEdge;
    }

    template <class Meta>
    inline Edge<Meta>::EdgePointer
    Edge<Meta>::getNextEdge() const noexcept
    {
      return m_nextEdge;
    }

    template <class Meta>
    inline Edge<Meta>::EdgePointer
    Edge<Meta>::getPreviousEdge() const noexcept
    {
      EdgePointer curEdge  = this;
      EdgePointer nextEdge = nullptr;

      while (nextEdge != this) {
        EdgePointer tmp = curEdge;

        curEdge  = nextEdge;
        nextEdge = curEdge->getNextEdge();
      }

      return curEdge;
    }

    template <class Meta>
    inline Edge<Meta>::FacePointer
    Edge<Meta>::getFace() const noexcept
    {
      return m_face;
    }

    template <class Meta>
    inline const Vec3&
    Edge<Meta>::getNormal() const noexcept
    {
      return (m_normal);
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
    inline Vec3
    Edge<Meta>::getX2X1() const noexcept
    {
      const auto& x1 = this->getVertex()->getPosition();
      const auto& x2 = this->getOtherVertex()->getPosition();

      return x2 - x1;
    }

    template <class Meta>
    inline Real
    Edge<Meta>::projectPointToEdge(const Vec3& a_x0) const noexcept
    {
      const auto p    = a_x0 - m_vertex->getPosition();
      const auto x2x1 = this->getX2X1();

      return p.dot(x2x1) / (x2x1.dot(x2x1));
    }

    template <class Meta>
    inline Real
    Edge<Meta>::signedDistance(const Vec3& a_x0) const noexcept
    {
      // Project point to edge
      const Real t = this->projectPointToEdge(a_x0);

      Real retval = 0.0;

      if (t <= 0.0) {
        // Closest point is the starting vertex.
        retval = this->getVertex()->signedDistance(a_x0);
      }
      else if (t >= 1.0) {
        retval = this->getOtherVertex()->signedDistance(a_x0);
      }
      else {
        // Closest point is the edge itself.
        const Vec3 x2x1      = this->getX2X1();
        const Vec3 linePoint = m_vertex->getPosition() + t * x2x1;
        const Vec3 delta     = a_x0 - linePoint;
        const Real dot       = m_normal.dot(delta);

        const int sgn = (dot > 0.0) ? 1 : -1;

        retval = sgn * delta.length();
      }

      return retval;
    }

    template <class Meta>
    inline Real
    Edge<Meta>::unsignedDistance2(const Vec3& a_x0) const noexcept
    {
      constexpr Real zero = 0.0;
      constexpr Real one  = 1.0;

      // Project point to edge and restrict to edge length.
      const auto t = std::min(std::max(zero, this->projectPointToEdge(a_x0)), one);

      // Compute distance to this edge.
      const Vec3 x2x1      = this->getX2X1();
      const Vec3 linePoint = m_vertex->getPosition() + t * x2x1;
      const Vec3 delta     = a_x0 - linePoint;

      return delta.dot(delta);
    }
  } // namespace DCEL
} // namespace EBGeometry

#include "EBGeometry_DCEL_VertexImplem.hpp"

#endif
