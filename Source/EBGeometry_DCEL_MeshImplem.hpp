/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_DCEL_MeshImplem.hpp
  @brief  Implementation of EBGeometry_DCEL_Mesh.hpp
  @author Robert Marskar
*/

#ifndef EBGeometry_DCEL_MeshImplem
#define EBGeometry_DCEL_MeshImplem

// Our includes
#include "EBGeometry_DCEL_Mesh.hpp"
#include "EBGeometry_Macros.hpp"

namespace EBGeometry {
  namespace DCEL {

    template <class Meta>
    inline Mesh<Meta>::Mesh() noexcept
    {
      m_numFaces    = -1;
      m_numEdges    = -1;
      m_numVertices = -1;

      m_faces    = nullptr;
      m_edges    = nullptr;
      m_vertices = nullptr;

      m_algorithm = SearchAlgorithm::Direct;
    }

    template <class Meta>
    inline Mesh<Meta>::Mesh(const int     a_numVertices,
                            const int     a_numEdges,
                            const int     a_numFaces,
                            Vertex<Meta>* a_vertices,
                            Edge<Meta>*   a_edges,
                            Face<Meta>*   a_faces) noexcept
      : Mesh<Meta>()
    {
      this->define(a_numVertices, a_numEdges, a_numFaces, a_vertices, a_edges, a_faces);
    }

    template <class Meta>
    inline Mesh<Meta>::~Mesh() noexcept
    {}

    template <class Meta>
    inline void
    Mesh<Meta>::define(const int     a_numVertices,
                       const int     a_numEdges,
                       const int     a_numFaces,
                       Vertex<Meta>* a_vertices,
                       Edge<Meta>*   a_edges,
                       Face<Meta>*   a_faces) noexcept
    {
      EBGEOMETRY_ALWAYS_EXPECT(a_numVertices >= 3);
      EBGEOMETRY_ALWAYS_EXPECT(a_numEdges >= 3);
      EBGEOMETRY_ALWAYS_EXPECT(a_numFaces >= 1);

      EBGEOMETRY_ALWAYS_EXPECT(a_vertices != nullptr);
      EBGEOMETRY_ALWAYS_EXPECT(a_edges != nullptr);
      EBGEOMETRY_ALWAYS_EXPECT(a_faces != nullptr);

      m_numVertices = a_numVertices;
      m_numEdges    = a_numEdges;
      m_numFaces    = a_numFaces;

      m_vertices = a_vertices;
      m_edges    = a_edges;
      m_faces    = a_faces;
    }

    template <class Meta>
    inline void
    Mesh<Meta>::sanityCheck() const noexcept
    {
#warning "Mesh<Meta>::sanityCheck() - not implemented"
    }

    template <class Meta>
    inline void
    Mesh<Meta>::setSearchAlgorithm(const SearchAlgorithm a_algorithm) noexcept
    {
      m_algorithm = a_algorithm;
    }

    template <class Meta>
    inline void
    Mesh<Meta>::setInsideOutsideAlgorithm(const Polygon2D::InsideOutsideAlgorithm a_algorithm) noexcept
    {
      EBGEOMETRY_EXPECT(m_numFaces > 0);
      EBGEOMETRY_EXPECT(m_faces != nullptr);

      for (int i = 0; i < m_numFaces; i++) {
        m_faces[i]->setInsideOutsideAlgorithm(a_algorithm);
      }
    }

    template <class Meta>
    inline void
    Mesh<Meta>::reconcile(const DCEL::VertexNormalWeight a_weight) noexcept
    {
      this->reconcileFaces();
      this->reconcileEdges();
      this->reconcileVertices(a_weight);
    }

    template <class Meta>
    inline const Vertex<Meta>*
    Mesh<Meta>::getVertices() const noexcept
    {
      return static_cast<const Vertex<Meta>*>(m_vertices);
    }

    template <class Meta>
    inline int
    Mesh<Meta>::getNumberOfVertices() const noexcept
    {
      return m_numVertices;
    }

    template <class Meta>
    inline const Edge<Meta>*
    Mesh<Meta>::getEdges() const noexcept
    {
      return static_cast<const Edge<Meta>*>(m_edges);
    }

    template <class Meta>
    inline int
    Mesh<Meta>::getNumberOfEdges() const noexcept
    {
      return m_numEdges;
    }

    template <class Meta>
    inline const Face<Meta>*
    Mesh<Meta>::getFaces() const noexcept
    {
      return static_cast<const Face<Meta>*>(m_faces);
    }

    template <class Meta>
    inline int
    Mesh<Meta>::getNumberOfFaces() const noexcept
    {
      return m_numFaces;
    }

    template <class Meta>
    inline Real
    Mesh<Meta>::signedDistance(const Vec3& a_point) const noexcept
    {
      return this->signedDistance(a_point, m_algorithm);
    }

    template <class Meta>
    inline Real
    Mesh<Meta>::signedDistance(const Vec3& a_point, SearchAlgorithm a_algorithm) const noexcept
    {
      Real dist = EBGeometry::Limits::max();

      switch (a_algorithm) {
      case SearchAlgorithm::Direct: {
        dist = this->DirectSignedDistance(a_point);

        break;
      }
      case SearchAlgorithm::Direct2: {
        dist = this->DirectSignedDistance2(a_point);

        break;
      }
      }

      return dist;
    }

    template <class Meta>
    inline Real
    Mesh<Meta>::unsignedDistance2(const Vec3& a_point) const noexcept
    {
      EBGEOMETRY_EXPECT(m_numFaces > 0);
      EBGEOMETRY_EXPECT(m_faces != nullptr);

      Real dist2 = EBGeometry::Limits::max();

      for (int i = 0; i < m_numFaces; i++) {
        const Real curDist2 = m_faces[i]->unsignedDistance2(a_point);

        dist2 = (curDist2 < dist2) ? curDist2 : dist2;
      }

      return dist2;
    }

    template <class Meta>
    inline void
    Mesh<Meta>::reconcileFaces() noexcept
    {
      EBGEOMETRY_EXPECT(m_numFaces > 0);
      EBGEOMETRY_EXPECT(m_faces != nullptr);

      for (int i = 0; i < m_numFaces; i++) {
        m_faces[i]->reconcile();
      }
    }

    template <class Meta>
    inline void
    Mesh<Meta>::reconcileEdges() noexcept
    {
      EBGEOMETRY_EXPECT(m_numEdges >= 3);
      EBGEOMETRY_EXPECT(m_edges != nullptr);

      for (int i = 0; i < m_numEdges; i++) {
        m_edges[i]->reconcile();
      }
    }

    template <class Meta>
    inline void
    Mesh<Meta>::reconcileVertices(const DCEL::VertexNormalWeight a_weight) noexcept
    {
      EBGEOMETRY_EXPECT(m_numVertices >= 3);
      EBGEOMETRY_EXPECT(m_vertices != nullptr);

      for (int i = 0; i < m_numVertices; i++) {
        switch (a_weight) {
        case DCEL::VertexNormalWeight::None: {
          m_vertices[i]->computeVertexNormalAverage();

          break;
        }
        case DCEL::VertexNormalWeight::Angle: {
          m_vertices[i]->computeVertexNormalAngleWeighted();

          break;
        }
        }
      }
    }

    template <class Meta>
    inline Real
    Mesh<Meta>::DirectSignedDistance(const Vec3& a_point) const noexcept
    {
      EBGEOMETRY_EXPECT(m_numFaces > 0);
      EBGEOMETRY_EXPECT(m_faces != nullptr);

      Real minDist  = EBGeometry::Limits::max();
      Real minDist2 = EBGeometry::Limits::max();

      for (int i = 0; i < m_numFaces; i++) {
        const Real curDist  = m_faces[i]->signedDistance(a_point);
        const Real curDist2 = curDist * curDist;

        if (curDist2 < minDist2) {
          minDist  = curDist;
          minDist2 = curDist2;
        }
      }

      return minDist;
    }

    template <class Meta>
    inline Real
    Mesh<Meta>::DirectSignedDistance2(const Vec3& a_point) const noexcept
    {
      EBGEOMETRY_EXPECT(m_numFaces > 0);
      EBGEOMETRY_EXPECT(m_faces != nullptr);

      int closest = -1;

      Real minDist2 = EBGeometry::Limits::max();

      for (int i = 0; i < m_numFaces; i++) {
        const Real curDist2 = m_faces[i]->unsignedDistance2(a_point);

        if (curDist2 < minDist2) {
          minDist2 = curDist2;
          closest  = i;
        }
      }

      EBGEOMETRY_EXPECT(closest >= 0);

      return m_faces[closest]->signedDistance(a_point);
    }

    template <class Meta>
    inline void
    Mesh<Meta>::incrementWarning(std::map<std::string, size_t>& a_warnings, const std::string& a_warn) const noexcept
    {}

    template <class Meta>
    inline void
    Mesh<Meta>::printWarnings(const std::map<std::string, size_t>& a_warnings) const noexcept
    {}
  } // namespace DCEL
} // namespace EBGeometry

#endif
