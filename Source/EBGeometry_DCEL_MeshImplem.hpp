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

// Std includes
#include <algorithm>
#include <vector>

// Our includes
#include "EBGeometry_DCEL_Mesh.hpp"
#include "EBGeometry_Macros.hpp"

namespace EBGeometry {
  namespace DCEL {

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE
    Mesh<Meta>::Mesh() noexcept
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
    EBGEOMETRY_ALWAYS_INLINE
    Mesh<Meta>::Mesh(const int     a_numVertices,
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
    EBGEOMETRY_ALWAYS_INLINE Mesh<Meta>::~Mesh() noexcept
    {}

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
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
    EBGEOMETRY_ALWAYS_INLINE void
    Mesh<Meta>::sanityCheck() const noexcept
    {
      const std::string vertexHasNoEdge            = "no referenced edge for vertex (unreferenced vertex)";
      const std::string vertexPositionIsDegenerate = "vertex position is degenerate (shares coordinate)";
      const std::string vertexHasNoVertexList      = "vertex has no vertex list";
      const std::string vertexHasNoEdgeList        = "vertex has no edge list";
      const std::string vertexHasNoFaceList        = "vertex has no face list";

      const std::string edgeIsCircular        = "edge start and end vertices are identical";
      const std::string edgeHasNoPairEdge     = "no pair edge (not watertight)";
      const std::string edgeHasNoNextEdge     = "no next edge (badly linked dcel)";
      const std::string edgeHasNoPreviousEdge = "no previous edge (badly linked dcel)";
      const std::string edgeHasNoStartVertex  = "no origin vertex found for half edge (badly linked dcel)";
      const std::string edgeHasNoEndVertex    = "no end vertex found for half edge (badly linked dcel)";
      const std::string edgeHasNoFace         = "no face found for half edge (badly linked dcel)";
      const std::string edgeHasBadPrevNext    = "previous edge's next edge is not this edge (badly linked dcel)";
      const std::string edgeHasBadNextPrev    = "next edge's previous edge is not this edge (badly linked dcel)";
      const std::string edgeHasNoVertexList   = "edge has no vertex list";
      const std::string edgeHasNoEdgeList     = "edge has no edge list";
      const std::string edgeHasNoFaceList     = "edge has no face list";

      const std::string faceIsNullptr       = "nullptr face";
      const std::string faceHasNoEdge       = "face has no edge";
      const std::string faceHasNoVertexList = "face has no vertex list";
      const std::string faceHasNoEdgeList   = "face has no edge list";
      const std::string faceHasNoFaceList   = "face has no face list";

      std::map<std::string, int> warnings = {{vertexHasNoEdge, 0},
                                             {vertexPositionIsDegenerate, 0},
                                             {edgeIsCircular, 0},
                                             {edgeHasNoStartVertex, 0},
                                             {edgeHasNoEndVertex, 0},
                                             {edgeHasNoPreviousEdge, 0},
                                             {edgeHasNoPairEdge, 0},
                                             {edgeHasNoNextEdge, 0},
                                             {edgeHasNoFace, 0},
                                             {edgeHasBadPrevNext, 0},
                                             {edgeHasBadNextPrev, 0},
                                             {faceHasNoEdge, 0}};

      // CHECK STANDARD ISSUES FOR VERTICES
      EBGEOMETRY_ALWAYS_EXPECT(m_numVertices >= 3);
      EBGEOMETRY_ALWAYS_EXPECT(m_vertices != nullptr);

      std::vector<Vec3> vertexCoordinates;

      for (int i = 0; i < m_numVertices; i++) {
        const Vertex<Meta>& curVertex = m_vertices[i];

        const int           edge       = curVertex.getEdge();
        const Vertex<Meta>* vertexList = curVertex.getVertexList();
        const Edge<Meta>*   edgeList   = curVertex.getEdgeList();
        const Face<Meta>*   faceList   = curVertex.getFaceList();

        if (edge < 0) {
          this->incrementWarning(warnings, vertexHasNoEdge);
        }
        if (vertexList == nullptr) {
          this->incrementWarning(warnings, vertexHasNoVertexList);
        }
        if (edgeList == nullptr) {
          this->incrementWarning(warnings, vertexHasNoEdgeList);
        }
        if (faceList == nullptr) {
          this->incrementWarning(warnings, vertexHasNoFaceList);
        }

        vertexCoordinates.push_back(m_vertices[i].getPosition());
      }

      // Sort vertices and vertex coordinates - then check for degenerate values.
      std::sort(std::begin(vertexCoordinates), std::end(vertexCoordinates), [](const Vec3& a, const Vec3& b) -> bool {
        return a.lessLX(b);
      });

      for (int i = 0; i < m_numVertices; i++) {
        if (vertexCoordinates[i] == vertexCoordinates[(i + 1) % m_numVertices]) {
          this->incrementWarning(warnings, vertexPositionIsDegenerate);
        }
      }

      // CHECK STANDARD ISSUES FOR EDGES
      EBGEOMETRY_ALWAYS_EXPECT(m_numEdges >= 3);
      EBGEOMETRY_ALWAYS_EXPECT(m_edges != nullptr);

      for (int i = 0; i < m_numEdges; i++) {
        const Edge<Meta>& curEdge = m_edges[i];

        const int startVertex  = curEdge.getVertex();
        const int endVertex    = curEdge.getOtherVertex();
        const int previousEdge = curEdge.getPreviousEdge();
        const int pairEdge     = curEdge.getPairEdge();
        const int nextEdge     = curEdge.getNextEdge();
        const int face         = curEdge.getFace();

        const Vertex<Meta>* vertexList = curEdge.getVertexList();
        const Edge<Meta>*   edgeList   = curEdge.getEdgeList();
        const Face<Meta>*   faceList   = curEdge.getFaceList();

        if (startVertex < 0) {
          this->incrementWarning(warnings, edgeHasNoStartVertex);
        }
        if (endVertex < 0) {
          this->incrementWarning(warnings, edgeHasNoEndVertex);
        }
        if (previousEdge < 0) {
          this->incrementWarning(warnings, edgeHasNoPreviousEdge);
        }
        if (pairEdge < 0) {
          this->incrementWarning(warnings, edgeHasNoPairEdge);
        }
        if (nextEdge < 0) {
          this->incrementWarning(warnings, edgeHasNoNextEdge);
        }
        if (face < 0) {
          this->incrementWarning(warnings, edgeHasNoFace);
        }
        if (startVertex == endVertex) {
          this->incrementWarning(warnings, edgeIsCircular);
        }
        if (vertexList == nullptr) {
          this->incrementWarning(warnings, edgeHasNoVertexList);
        }
        if (edgeList == nullptr) {
          this->incrementWarning(warnings, edgeHasNoEdgeList);
        }
        if (faceList == nullptr) {
          this->incrementWarning(warnings, edgeHasNoFaceList);
        }

        // More nuanced issues
        if (previousEdge > 0) {
          if (m_edges[previousEdge].getNextEdge() != i) {
            this->incrementWarning(warnings, edgeHasBadNextPrev);
          }
        }
        if (nextEdge > 0) {
          if (m_edges[nextEdge].getPreviousEdge() != i) {
            this->incrementWarning(warnings, edgeHasBadPrevNext);
          }
        }
      }

      // CHECK STANDARD ISSUES FOR FACES
      EBGEOMETRY_ALWAYS_EXPECT(m_numFaces >= 1);
      EBGEOMETRY_ALWAYS_EXPECT(m_faces != nullptr);

      for (int i = 0; i < m_numFaces; i++) {
        const Face<Meta>& curFace = m_faces[i];

        const int edge = curFace.getEdge();

        const Vertex<Meta>* vertexList = curFace.getVertexList();
        const Edge<Meta>*   edgeList   = curFace.getEdgeList();
        const Face<Meta>*   faceList   = curFace.getFaceList();

        if (edge < 0) {
          this->incrementWarning(warnings, faceHasNoEdge);
        }

        if (vertexList == nullptr) {
          this->incrementWarning(warnings, edgeHasNoVertexList);
        }
        if (edgeList == nullptr) {
          this->incrementWarning(warnings, edgeHasNoEdgeList);
        }
        if (faceList == nullptr) {
          this->incrementWarning(warnings, edgeHasNoFaceList);
        }
      }

      this->printWarnings(warnings);
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Mesh<Meta>::setSearchAlgorithm(const SearchAlgorithm a_algorithm) noexcept
    {
      m_algorithm = a_algorithm;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Mesh<Meta>::setInsideOutsideAlgorithm(const Polygon2D::InsideOutsideAlgorithm a_algorithm) noexcept
    {
      EBGEOMETRY_EXPECT(m_numFaces > 0);
      EBGEOMETRY_EXPECT(m_faces != nullptr);

      for (int i = 0; i < m_numFaces; i++) {
        m_faces[i].setInsideOutsideAlgorithm(a_algorithm);
      }
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Mesh<Meta>::reconcile(const DCEL::VertexNormalWeight a_weight) noexcept
    {
      this->reconcileFaces();
      this->reconcileEdges();
      this->reconcileVertices(a_weight);
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE const Vertex<Meta>*
                                   Mesh<Meta>::getVertices() const noexcept
    {
      return static_cast<const Vertex<Meta>*>(m_vertices);
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE int
    Mesh<Meta>::getNumberOfVertices() const noexcept
    {
      return m_numVertices;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE const Edge<Meta>*
                                   Mesh<Meta>::getEdges() const noexcept
    {
      return static_cast<const Edge<Meta>*>(m_edges);
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE int
    Mesh<Meta>::getNumberOfEdges() const noexcept
    {
      return m_numEdges;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE const Face<Meta>*
                                   Mesh<Meta>::getFaces() const noexcept
    {
      return static_cast<const Face<Meta>*>(m_faces);
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE int
    Mesh<Meta>::getNumberOfFaces() const noexcept
    {
      return m_numFaces;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE Real
    Mesh<Meta>::signedDistance(const Vec3& a_point) const noexcept
    {
      return this->signedDistance(a_point, m_algorithm);
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE Real
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
    EBGEOMETRY_ALWAYS_INLINE Real
    Mesh<Meta>::unsignedDistance2(const Vec3& a_point) const noexcept
    {
      EBGEOMETRY_EXPECT(m_numFaces > 0);
      EBGEOMETRY_EXPECT(m_faces != nullptr);

      Real dist2 = EBGeometry::Limits::max();

      for (int i = 0; i < m_numFaces; i++) {
        const Real curDist2 = m_faces[i].unsignedDistance2(a_point);

        dist2 = (curDist2 < dist2) ? curDist2 : dist2;
      }

      return dist2;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Mesh<Meta>::reconcileFaces() noexcept
    {
      EBGEOMETRY_EXPECT(m_numFaces > 0);
      EBGEOMETRY_EXPECT(m_faces != nullptr);

      for (int i = 0; i < m_numFaces; i++) {
        m_faces[i].reconcile();
      }
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Mesh<Meta>::reconcileEdges() noexcept
    {
      EBGEOMETRY_EXPECT(m_numEdges >= 3);
      EBGEOMETRY_EXPECT(m_edges != nullptr);

      for (int i = 0; i < m_numEdges; i++) {
        m_edges[i].reconcile();
      }
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Mesh<Meta>::reconcileVertices(const DCEL::VertexNormalWeight a_weight) noexcept
    {
      EBGEOMETRY_EXPECT(m_numVertices >= 3);
      EBGEOMETRY_EXPECT(m_vertices != nullptr);

      for (int i = 0; i < m_numVertices; i++) {
        switch (a_weight) {
        case DCEL::VertexNormalWeight::None: {
          m_vertices[i].computeVertexNormalAverage();

          break;
        }
        case DCEL::VertexNormalWeight::Angle: {
          m_vertices[i].computeVertexNormalAngleWeighted();

          break;
        }
        }
      }
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE Real
    Mesh<Meta>::DirectSignedDistance(const Vec3& a_point) const noexcept
    {
      EBGEOMETRY_EXPECT(m_numFaces > 0);
      EBGEOMETRY_EXPECT(m_faces != nullptr);

      Real minDist  = EBGeometry::Limits::max();
      Real minDist2 = EBGeometry::Limits::max();

      for (int i = 0; i < m_numFaces; i++) {
        const Real curDist  = m_faces[i].signedDistance(a_point);
        const Real curDist2 = curDist * curDist;

        if (curDist2 < minDist2) {
          minDist  = curDist;
          minDist2 = curDist2;
        }
      }

      return minDist;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE Real
    Mesh<Meta>::DirectSignedDistance2(const Vec3& a_point) const noexcept
    {
      EBGEOMETRY_EXPECT(m_numFaces > 0);
      EBGEOMETRY_EXPECT(m_faces != nullptr);

      int closest = -1;

      Real minDist2 = EBGeometry::Limits::max();

      for (int i = 0; i < m_numFaces; i++) {
        const Real curDist2 = m_faces[i].unsignedDistance2(a_point);

        if (curDist2 < minDist2) {
          minDist2 = curDist2;
          closest  = i;
        }
      }

      EBGEOMETRY_EXPECT(closest >= 0);

      return m_faces[closest].signedDistance(a_point);
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Mesh<Meta>::incrementWarning(std::map<std::string, int>& a_warnings, const std::string& a_warn) const noexcept
    {
      a_warnings.at(a_warn) += 1;
    }

    template <class Meta>
    EBGEOMETRY_ALWAYS_INLINE void
    Mesh<Meta>::printWarnings(const std::map<std::string, int>& a_warnings) const noexcept
    {
      for (const auto& warn : a_warnings) {
        if (warn.second > 0) {
          std::cerr << "In file 'EBGeometry_DCEL_MeshImplem.H' function "
                       "Mesh<Meta>::sanityCheck() - warnings about error '"
                    << warn.first << "' = " << warn.second << "\n";
        }
      }
    }
  } // namespace DCEL
} // namespace EBGeometry

#endif
