/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_DCEL_FaceImplem.hpp
  @brief  Implementation of EBGeometry_DCEL_Face.hpp
  @author Robert Marskar
*/

#ifndef EBGeometry_DCEL_FaceImplem
#define EBGeometry_DCEL_FaceImplem

// Our includes
#include "EBGeometry_DCEL_Face.hpp"
#include "EBGeometry_Macros.hpp"

namespace EBGeometry::DCEL {

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  Face<MetaData>::Face(int a_edge) noexcept :
    m_edge(a_edge)
  {}

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  void
  Face<MetaData>::reconcile() noexcept
  {
    this->computeNormal();
    this->normalizeNormalVector();
    this->computeCentroid();
    this->computePolygon2D();
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  void
  Face<MetaData>::setEdge(int a_edge) noexcept
  {
    m_edge = a_edge;
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  void
  Face<MetaData>::setMetaData(const MetaData& a_metaData) noexcept
  {
    m_metaData = a_metaData;
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  void
  Face<MetaData>::setVertexList(const Vertex<MetaData>* const a_vertexList) noexcept
  {
    m_vertexList = a_vertexList;
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  void
  Face<MetaData>::setEdgeList(const Edge<MetaData>* const a_edgeList) noexcept
  {
    m_edgeList = a_edgeList;
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  void
  Face<MetaData>::setFaceList(const Face<MetaData>* const a_faceList) noexcept
  {
    m_faceList = a_faceList;
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  const Vertex<MetaData>*
  Face<MetaData>::getVertexList() const noexcept
  {
    return (m_vertexList);
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  const Edge<MetaData>*
  Face<MetaData>::getEdgeList() const noexcept
  {
    return (m_edgeList);
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  const Face<MetaData>*
  Face<MetaData>::getFaceList() const noexcept
  {
    return (m_faceList);
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  void
  Face<MetaData>::setInsideOutsideAlgorithm(const Polygon2D::InsideOutsideAlgorithm& a_algorithm) noexcept
  {
    m_poly2Algorithm = a_algorithm;
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  int
  Face<MetaData>::getNumEdges() const noexcept
  {
    EBGEOMETRY_EXPECT(m_edge >= 0);
    EBGEOMETRY_EXPECT(m_vertexList != nullptr);
    EBGEOMETRY_EXPECT(m_edgeList != nullptr);
    EBGEOMETRY_EXPECT(m_faceList != nullptr);

    int numEdges = 0;
    int curEdge  = -1;

    while (curEdge != m_edge) {
      curEdge = (curEdge < 0) ? m_edge : curEdge;

      EBGEOMETRY_EXPECT(curEdge >= 0);

      numEdges += 1;

      curEdge = m_edgeList[curEdge].getNextEdge();

      EBGEOMETRY_EXPECT(curEdge >= 0);
    }

    return numEdges;
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  int
  Face<MetaData>::getEdge() const noexcept
  {
    return m_edge;
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  Vec3&
  Face<MetaData>::getCentroid() noexcept
  {
    return (m_centroid);
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  const Vec3&
  Face<MetaData>::getCentroid() const noexcept
  {
    return (m_centroid);
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  Real&
  Face<MetaData>::getCentroid(int a_dir) noexcept
  {
    EBGEOMETRY_EXPECT(a_dir >= 0);
    EBGEOMETRY_EXPECT(a_dir < 3);

    return (m_centroid[a_dir]);
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  const Real&
  Face<MetaData>::getCentroid(int a_dir) const noexcept
  {
    EBGEOMETRY_EXPECT(a_dir >= 0);
    EBGEOMETRY_EXPECT(a_dir < 3);

    return (m_centroid[a_dir]);
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  Vec3&
  Face<MetaData>::getNormal() noexcept
  {
    return (m_normal);
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  const Vec3&
  Face<MetaData>::getNormal() const noexcept
  {
    return (m_normal);
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  MetaData&
  Face<MetaData>::getMetaData() noexcept
  {
    return (m_metaData);
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  const MetaData&
  Face<MetaData>::getMetaData() const noexcept
  {
    return (m_metaData);
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  Real
  Face<MetaData>::signedDistance(const Vec3& a_x0) const noexcept
  {
    EBGEOMETRY_EXPECT(m_edge >= 0);
    EBGEOMETRY_EXPECT(m_vertexList != nullptr);
    EBGEOMETRY_EXPECT(m_edgeList != nullptr);
    EBGEOMETRY_EXPECT(m_faceList != nullptr);

    Real minDist = EBGeometry::Limits::max();

    const bool inside = this->isPointInsideFace(a_x0);

    if (inside) {
      minDist = m_normal.dot(a_x0 - m_centroid);
    }
    else {
      int curEdge = -1;

      while (curEdge != m_edge) {
        curEdge = (curEdge < 0) ? m_edge : curEdge;

        const Real curDist = m_edgeList[curEdge].signedDistance(a_x0);

        minDist = (std::abs(curDist) < std::abs(minDist)) ? curDist : minDist;

        // Go to next edge and exit if we've circulated all half-edges
        // in this face.
        curEdge = m_edgeList[curEdge].getNextEdge();
        EBGEOMETRY_EXPECT(curEdge >= 0);
      }
    }

    return minDist;
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  Real
  Face<MetaData>::unsignedDistance2(const Vec3& a_x0) const noexcept
  {
    EBGEOMETRY_EXPECT(m_edge >= 0);
    EBGEOMETRY_EXPECT(m_vertexList != nullptr);
    EBGEOMETRY_EXPECT(m_edgeList != nullptr);
    EBGEOMETRY_EXPECT(m_faceList != nullptr);

    Real minDist2 = EBGeometry::Limits::max();

    const bool inside = this->isPointInsideFace(a_x0);

    if (inside) {
      const Real curDist = m_normal.dot(a_x0 - m_centroid);

      minDist2 = curDist * curDist;
    }
    else {
      int curEdge = -1;
      while (curEdge != m_edge) {
        curEdge = (curEdge < 0) ? m_edge : curEdge;

        const Real curDist2 = m_edgeList[curEdge].unsignedDistance2(a_x0);

        minDist2 = (std::abs(curDist2) < std::abs(minDist2)) ? curDist2 : minDist2;

        // Go to next edge and exit if we've circulated all half-edges
        // in this face.
        curEdge = m_edgeList[curEdge].getNextEdge();

        EBGEOMETRY_EXPECT(curEdge >= 0);
      }
    }

    return minDist2;
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  Vec3
  Face<MetaData>::getSmallestCoordinate() const noexcept
  {
    EBGEOMETRY_EXPECT(m_edge >= 0);
    EBGEOMETRY_EXPECT(m_vertexList != nullptr);
    EBGEOMETRY_EXPECT(m_edgeList != nullptr);
    EBGEOMETRY_EXPECT(m_faceList != nullptr);

    Vec3 smallestCoordinate = Vec3::max();

    int curEdge = -1;

    while (curEdge != m_edge) {
      curEdge = (curEdge < 0) ? m_edge : curEdge;

      EBGEOMETRY_EXPECT(curEdge >= 0);

      const int vertex = m_edgeList[curEdge].getVertex();

      EBGEOMETRY_EXPECT(vertex >= 0);

      smallestCoordinate = min(smallestCoordinate, m_vertexList[vertex].getPosition());

      curEdge = m_edgeList[curEdge].getNextEdge();

      EBGEOMETRY_EXPECT(curEdge >= 0);
    }

    return smallestCoordinate;
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  Vec3
  Face<MetaData>::getHighestCoordinate() const noexcept
  {
    EBGEOMETRY_EXPECT(m_edge >= 0);
    EBGEOMETRY_EXPECT(m_vertexList != nullptr);
    EBGEOMETRY_EXPECT(m_edgeList != nullptr);
    EBGEOMETRY_EXPECT(m_faceList != nullptr);

    Vec3 highestCoordinate = Vec3::lowest();

    int curEdge = -1;

    while (curEdge != m_edge) {
      curEdge = (curEdge < 0) ? m_edge : curEdge;

      EBGEOMETRY_EXPECT(curEdge >= 0);

      const int vertex = m_edgeList[curEdge].getVertex();

      EBGEOMETRY_EXPECT(vertex >= 0);

      highestCoordinate = max(highestCoordinate, m_vertexList[vertex].getPosition());

      curEdge = m_edgeList[curEdge].getNextEdge();

      EBGEOMETRY_EXPECT(curEdge >= 0);
    }

    return highestCoordinate;
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  void
  Face<MetaData>::computeCentroid() noexcept
  {
    EBGEOMETRY_EXPECT(m_edge >= 0);
    EBGEOMETRY_EXPECT(m_vertexList != nullptr);
    EBGEOMETRY_EXPECT(m_edgeList != nullptr);
    EBGEOMETRY_EXPECT(m_faceList != nullptr);

    m_centroid = Vec3::zero();

    int numVertices = 0;
    int curEdge     = -1;

    while (curEdge != m_edge) {
      curEdge = (curEdge < 0) ? m_edge : curEdge;

      EBGEOMETRY_EXPECT(curEdge >= 0);

      const int vertex = m_edgeList[curEdge].getVertex();

      EBGEOMETRY_EXPECT(vertex >= 0);

      m_centroid += m_vertexList[vertex].getPosition();

      curEdge     = m_edgeList[curEdge].getNextEdge();
      numVertices = numVertices + 1;

      EBGEOMETRY_EXPECT(curEdge >= 0);
    }

    EBGEOMETRY_EXPECT(numVertices > 0);

    m_centroid = m_centroid / Real(numVertices);
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  void
  Face<MetaData>::computeNormal() noexcept
  {
    EBGEOMETRY_EXPECT(m_edge >= 0);
    EBGEOMETRY_EXPECT(m_vertexList != nullptr);
    EBGEOMETRY_EXPECT(m_edgeList != nullptr);
    EBGEOMETRY_EXPECT(m_faceList != nullptr);

    // Circulate through the polygon and find three vertices that do
    // not lie on a line. Use their coordinates to find the normal vector
    // that is orthogonal to the plane that they span.
    m_normal = Vec3::zero();

    int curEdge = -1;

    while (curEdge != m_edge) {
      curEdge = (curEdge < 0) ? m_edge : curEdge;

      EBGEOMETRY_EXPECT(curEdge >= 0);

      const int nextEdge = m_edgeList[curEdge].getNextEdge();

      EBGEOMETRY_EXPECT(nextEdge >= 0);

      const int nextNextEdge = m_edgeList[nextEdge].getNextEdge();

      EBGEOMETRY_EXPECT(nextNextEdge >= 0);

      EBGEOMETRY_EXPECT(curEdge != nextEdge);
      EBGEOMETRY_EXPECT(nextEdge != nextNextEdge);
      EBGEOMETRY_EXPECT(nextNextEdge != curEdge);

      const int v0 = m_edgeList[curEdge].getVertex();
      const int v1 = m_edgeList[nextEdge].getVertex();
      const int v2 = m_edgeList[nextNextEdge].getVertex();

      EBGEOMETRY_EXPECT(v0 >= 0);
      EBGEOMETRY_EXPECT(v1 >= 0);
      EBGEOMETRY_EXPECT(v2 >= 0);

      EBGEOMETRY_EXPECT(v0 != v1);
      EBGEOMETRY_EXPECT(v1 != v2);
      EBGEOMETRY_EXPECT(v2 != v0);

      const Vec3& x0 = m_vertexList[v0].getPosition();
      const Vec3& x1 = m_vertexList[v1].getPosition();
      const Vec3& x2 = m_vertexList[v2].getPosition();

      m_normal = (x2 - x1).cross(x2 - x0);

      // Length of the normal vector will be > 0.0 when the vertices do not lie on a line.
      if (m_normal.length() > Real(0.0)) {
        break;
      }

      // Go to next edge
      curEdge = m_edgeList[curEdge].getNextEdge();

      EBGEOMETRY_EXPECT(curEdge >= 0);
    }

    EBGEOMETRY_EXPECT(m_normal.length() > EBGeometry::Limits::min());

    this->normalizeNormalVector();
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  void
  Face<MetaData>::computePolygon2D() noexcept
  {
    EBGEOMETRY_EXPECT(m_edge >= 0);
    EBGEOMETRY_EXPECT(m_vertexList != nullptr);
    EBGEOMETRY_EXPECT(m_edgeList != nullptr);
    EBGEOMETRY_EXPECT(m_faceList != nullptr);

    const int numEdges = this->getNumEdges();

    EBGEOMETRY_EXPECT(numEdges >= 3);

    // clang-tidy wants us to handle exceptions here, but there can
    // be no exceptions in device code.
    Vec3* vertexCoordinates = new Vec3[numEdges]; // NOLINT

    int curEdge = -1;
    int counter = 0;

    while (curEdge != m_edge) {
      curEdge = (curEdge < 0) ? m_edge : curEdge;

      EBGEOMETRY_EXPECT(curEdge >= 0);

      const int vertex = m_edgeList[curEdge].getVertex();

      EBGEOMETRY_EXPECT(vertex >= 0);

      const Vec3& x = m_vertexList[vertex].getPosition();

      vertexCoordinates[counter] = x;

      curEdge = m_edgeList[curEdge].getNextEdge();

      EBGEOMETRY_EXPECT(curEdge >= 0);

      counter++;
    }

    m_polygon2D.define(m_normal, counter, vertexCoordinates);

    delete[] vertexCoordinates;
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  void
  Face<MetaData>::normalizeNormalVector() noexcept
  {
    EBGEOMETRY_EXPECT(m_normal.length() > EBGeometry::Limits::min());

    m_normal = m_normal / m_normal.length();
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  Real
  Face<MetaData>::computeArea() noexcept
  {
    EBGEOMETRY_EXPECT(m_edge >= 0);
    EBGEOMETRY_EXPECT(m_vertexList != nullptr);
    EBGEOMETRY_EXPECT(m_edgeList != nullptr);
    EBGEOMETRY_EXPECT(m_faceList != nullptr);

    Real area = 0.0;

    int curEdge = -1;

    while (curEdge != m_edge) {
      curEdge = (curEdge < 0) ? m_edge : curEdge;

      EBGEOMETRY_EXPECT(curEdge >= 0);

      const int nextEdge = m_edgeList[curEdge].getNextEdge();

      EBGEOMETRY_EXPECT(nextEdge >= 0);

      const int v0 = m_edgeList[curEdge].getVertex();
      const int v1 = m_edgeList[nextEdge].getVertex();

      EBGEOMETRY_EXPECT(v0 >= 0);
      EBGEOMETRY_EXPECT(v1 >= 0);
      EBGEOMETRY_EXPECT(v0 != v1);

      const Vec3& x0 = m_vertexList[v0].getPosition();
      const Vec3& x1 = m_vertexList[v1].getPosition();

      area += m_normal.dot(x1.cross(x0));

      curEdge = m_edgeList[curEdge].getNextEdge();

      EBGEOMETRY_EXPECT(curEdge >= 0);
    }

    area = Real(0.5) * std::abs(area);

    return area;
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  Vec3
  Face<MetaData>::projectPointIntoFacePlane(const Vec3& a_p) const noexcept
  {
    return a_p - m_normal * (m_normal.dot(a_p - m_centroid));
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  bool
  Face<MetaData>::isPointInsideFace(const Vec3& a_p) const noexcept
  {
    const Vec3 p = this->projectPointIntoFacePlane(a_p);

    return m_polygon2D.isPointInside(p, m_poly2Algorithm);
  }
} // namespace EBGeometry::DCEL

#endif
