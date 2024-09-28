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

namespace EBGeometry {
  namespace DCEL {

    template <class Meta>
    inline Face<Meta>::Face() noexcept
    {
      m_halfEdge  = nullptr;
      m_normal    = Vec3::zero();
      m_centroid  = Vec3::zero();
      m_polygon2D = Polygon2D();
    }

    template <class Meta>
    inline Face<Meta>::Face(const EdgePointer a_edge) noexcept : Face()
    {
      m_halfEdge = a_edge;
    }

    template <class Meta>
    inline Face<Meta>::Face(const Face& a_otherFace) noexcept : Face()
    {
      m_halfEdge = a_otherFace.m_halfEdge;
      m_normal   = a_otherFace.m_normal;
    }

    template <class Meta>
    inline Face<Meta>::~Face() noexcept
    {}

    template <class Meta>
    inline void
    Face<Meta>::define(const Vec3& a_normal, const EdgePointer a_edge) noexcept
    {
      m_normal   = a_normal;
      m_halfEdge = a_edge;
    }

    template <class Meta>
    inline void
    Face<Meta>::reconcile() noexcept
    {
      this->computeNormalVector();
      this->normalizeNormalVector();
      this->computeCentroid();
      this->computeArea();
      this->computePolygon2D();
    }

    template <class Meta>
    inline void
    Face<Meta>::setHalfEdge(const EdgePointer& a_halfEdge) noexcept
    {
      m_halfEdge = a_halfEdge;
    }

    template <class Meta>
    inline void
    Face<Meta>::setInsideOutsideAlgorithm(const Polygon2D::InsideOutsideAlgorithm& a_algorithm) noexcept
    {
      m_poly2Algorithm = a_algorithm;
    }

    template <class Meta>
    inline Face<Meta>::EdgePointer
    Face<Meta>::getHalfEdge() const noexcept
    {
      return m_halfEdge;
    }

    template <class Meta>
    inline Vec3&
    Face<Meta>::getCentroid() noexcept
    {
      return (m_centroid);
    }

    template <class Meta>
    inline const Vec3&
    Face<Meta>::getCentroid() const noexcept
    {
      return (m_centroid);
    }

    template <class Meta>
    inline Real&
    Face<Meta>::getCentroid(const int a_dir) noexcept
    {
      EBGEOMETRY_EXPECT(a_dir >= 0);
      EBGEOMETRY_EXPECT(a_dir < 3);

      return (m_centroid[a_dir]);
    }

    template <class Meta>
    inline const Real&
    Face<Meta>::getCentroid(const int a_dir) const noexcept
    {
      EBGEOMETRY_EXPECT(a_dir >= 0);
      EBGEOMETRY_EXPECT(a_dir < 3);

      return (m_centroid[a_dir]);
    }

    template <class Meta>
    inline Vec3&
    Face<Meta>::getNormal() noexcept
    {
      return (m_normal);
    }

    template <class Meta>
    inline const Vec3&
    Face<Meta>::getNormal() const noexcept
    {
      return (m_normal);
    }

    template <class Meta>
    inline Meta&
    Face<Meta>::getMetaData() noexcept
    {
      return (m_metaData);
    }

    template <class Meta>
    inline const Meta&
    Face<Meta>::getMetaData() const noexcept
    {
      return (m_metaData);
    }

    template <class Meta>
    inline Real
    Face<Meta>::signedDistance(const Vec3& a_x0) const noexcept
    {
      Real minDist = EBGeometry::Limits::max();

      const bool inside = this->isPointInside(a_x0);

      if(inside) {
	minDist = m_normal.dot(a_x0 - m_centroid);
      }
      else {
	EdgePointer curEdge = m_halfEdge;

	while(true) {
	  const Real curDist = curEdge->signedDistance(a_x0);

	  minDist = (std::abs(curDist) < std::abs(minDist))? curDist : minDist;

	  // Go to next edge and exit if we've circulated all half-edges
	  // in this face. 
	  curEdge = curEdge->getNextEdge();

	  EBGEOMETRY_EXPECT(curEdge != nullptr);
	  
	  if(curEdge == m_halfEdge) {
	    break;
	  }
	}
      }

      return minDist;
    }

    template <class Meta>
    inline Real
    Face<Meta>::unsignedDistance2(const Vec3& a_x0) const noexcept
    {
      Real minDist2 = EBGeometry::Limits::max();

      const bool inside = this->isPointInside(a_x0);

      if(inside) {
	const Real curDist = m_normal.dot(a_x0 - m_centroid);
	
	minDist2 = curDist * curDist;
      }
      else {
	EdgePointer curEdge = m_halfEdge;

	while(true) {
	  const Real curDist2 = curEdge->unsignedDistance2(a_x0);

	  minDist2 = (std::abs(curDist2) < std::abs(minDist2))? curDist2 : minDist2;

	  // Go to next edge and exit if we've circulated all half-edges
	  // in this face. 
	  curEdge = curEdge->getNextEdge();

	  EBGEOMETRY_EXPECT(curEdge != nullptr);
	  
	  if(curEdge == m_halfEdge) {
	    break;
	  }
	}
      }

      return minDist2;      
    }

    template <class Meta>
    inline Vec3
    Face<Meta>::getSmallestCoordinate() const noexcept
    {
      Vec3 smallestCoordinate = Vec3::max();

      EBGEOMETRY_EXPECT(m_halfEdge != nullptr);

      EdgePointer curEdge = nullptr;

      while (curEdge != m_halfEdge) {
        curEdge = (curEdge == nullptr) ? m_halfEdge : curEdge;

        EBGEOMETRY_EXPECT(curEdge != nullptr);

        VertexPointer vertex = curEdge->getVertex();

        EBGEOMETRY_EXPECT(vertex != nullptr);

        smallestCoordinate = min(smallestCoordinate, vertex->getPosition());

        curEdge = curEdge->getNextEdge();
      }

      return smallestCoordinate;
    }

    template <class Meta>
    inline Vec3
    Face<Meta>::getHighestCoordinate() const noexcept
    {
      Vec3 highestCoordinate = Vec3::lowest();

      EBGEOMETRY_EXPECT(m_halfEdge != nullptr);

      EdgePointer curEdge = nullptr;

      while (curEdge != m_halfEdge) {
        curEdge = (curEdge == nullptr) ? m_halfEdge : curEdge;

        EBGEOMETRY_EXPECT(curEdge != nullptr);

        VertexPointer vertex = curEdge->getVertex();

        EBGEOMETRY_EXPECT(vertex != nullptr);

        highestCoordinate = max(highestCoordinate, vertex->getPosition());

        curEdge = curEdge->getNextEdge();
      }

      return highestCoordinate;
    }

    template <class Meta>
    inline void
    Face<Meta>::computeCentroid() noexcept
    {
      m_centroid = Vec3::zero();

      int numEdges = 0;

      EdgePointer curEdge = nullptr;

      while (curEdge != m_halfEdge) {
        curEdge = (curEdge == nullptr) ? m_halfEdge : curEdge;

        EBGEOMETRY_EXPECT(curEdge != nullptr);

        VertexPointer vertex = curEdge->getVertex();

        EBGEOMETRY_EXPECT(vertex != nullptr);

        m_centroid += vertex->getPosition();

        curEdge  = curEdge->getNextEdge();
        numEdges = numEdges + 1;
      }

      EBGEOMETRY_EXPECT(numEdges > 0);

      m_centroid = m_centroid / numEdges;
    }

    template <class Meta>
    inline void
    Face<Meta>::computeNormal() noexcept
    {
      // Circulate through the polygon and find three vertices that do
      // not lie on a line. Use their coordinates to find the normal vector
      // that is orthogonal to the plane that they span.
      m_normal = Vec3::zero();

      EdgePointer curEdge = nullptr;

      while (curEdge != m_halfEdge) {
        curEdge = (curEdge == nullptr) ? m_halfEdge : curEdge;

        EBGEOMETRY_EXPECT(curEdge != nullptr);
        EBGEOMETRY_EXPECT(curEdge->getNextEdge() != nullptr);
        EBGEOMETRY_EXPECT(curEdge->getNextEdge()->getNextEdge() != nullptr);

        VertexPointer v0 = curEdge->getVertex();
        VertexPointer v1 = curEdge->getNextEdge()->getVertex();
        VertexPointer v2 = curEdge->getNextEdge()->getNextEdge()->getVertex();

        EBGEOMETRY_EXPECT(v0 != nullptr);
        EBGEOMETRY_EXPECT(v1 != nullptr);
        EBGEOMETRY_EXPECT(v2 != nullptr);

        const Vec3 x0 = v0->getPosition();
        const Vec3 x1 = v1->getPosition();
        const Vec3 x2 = v2->getPosition();

        m_normal = (x2 - x1).cross(x2 - x0);

        // Length of the normal vector will be > 0.0 when the vertices do not lie on a line.
        if (m_normal.length() > Real(0.0)) {
          break;
        }

        // Go to next edge
        curEdge = curEdge->getNextEdge();
      }

      EBGEOMETRY_EXPECT(m_normal.length() > EBGeometry::Limits::min());

      this->normalizeNormalVector();
    }

    template <class Meta>
    inline void
    Face<Meta>::computePolygon2D() noexcept
    {

      // Figure out number of vertices in this face.
      int         counter = 0;
      EdgePointer curEdge = nullptr;

      while (curEdge != m_halfEdge) {
        curEdge = (curEdge == nullptr) ? m_halfEdge : curEdge;

        EBGEOMETRY_EXPECT(curEdge != nullptr);

        counter = counter + 1;

        curEdge = curEdge->getNextEdge();
      }

      // Get the vertex coordinates.
      Vec3* vertexCoordinates = new Vec3[counter];

      curEdge = nullptr;
      counter = 0;

      while (curEdge != m_halfEdge) {
        curEdge = (curEdge == nullptr) ? m_halfEdge : curEdge;

        EBGEOMETRY_EXPECT(curEdge != nullptr);
        EBGEOMETRY_EXPECT(curEdge->getVertex() != nullptr);

        vertexCoordinates[counter] = curEdge->getVertex()->getPosition();

        curEdge = curEdge->getNextEdge();
        counter = counter + 1;
      }

      m_polygon2D.define(m_normal, counter, vertexCoordinates);

      delete[] vertexCoordinates;
    }

    template <class Meta>
    inline void
    Face<Meta>::normalizeNormalVector() noexcept
    {
      EBGEOMETRY_EXPECT(m_normal.length() > Real(0.0));

      m_normal = m_normal / m_normal.length();
    }

    template <class Meta>
    inline Real
    Face<Meta>::computeArea() noexcept
    {
      Real area = 0.0;

      EdgePointer curEdge = nullptr;

      while (curEdge != m_halfEdge) {
        curEdge = (curEdge == nullptr) ? m_halfEdge : curEdge;

        EBGEOMETRY_EXPECT(curEdge != nullptr);
        EBGEOMETRY_EXPECT(curEdge->getNextEdge() != nullptr);

        VertexPointer v0 = curEdge->getVertex();
        VertexPointer v1 = curEdge->getNextEdge()->getVertex();

        EBGEOMETRY_EXPECT(v0 != nullptr);
        EBGEOMETRY_EXPECT(v1 != nullptr);

        const Vec3 x0 = v0->getPosition();
        const Vec3 x1 = v1->getPosition();

        area += m_normal.dot(x1.cross(x0));

        curEdge = curEdge->getNextEdge();
      }

      area = 0.5 * std::abs(area);

      return area;
    }

    template <class Meta>
    inline Vec3
    Face<Meta>::projectPointIntoFacePlane(const Vec3& a_p) const noexcept
    {
      return a_p - m_normal * (m_normal.dot(a_p - m_centroid));
    }

    template <class Meta>
    inline bool
    Face<Meta>::isPointInsideFace(const Vec3& a_p) const noexcept
    {
      const Vec3 p = this->projectPointIntoFacePlane(a_p);

      return m_polygon2D.isPointInside(p, m_poly2Algorithm);
    }
  } // namespace DCEL
} // namespace EBGeometry

#endif
