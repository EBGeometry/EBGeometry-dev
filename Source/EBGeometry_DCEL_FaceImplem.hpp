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
      m_halfEdge = nullptr;
      m_normal   = Vec3::zero();
      m_centroid = Vec3::zero();
      m_poly2    = Polygon2D();
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
    inline EdgePointer
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

      return (m_centroid);
    }

    template <class Meta>
    inline const Real&
    Face<Meta>::getCentroid(const int a_dir) const noexcept
    {
      EBGEOMETRY_EXPECT(a_dir >= 0);
      EBGEOMETRY_EXPECT(a_dir < 3);

      return (m_centroid);
    }

    template <class Meta>
    inline Real&
    Face<Meta>::getArea() noexcept
    {
      return (m_area);
    }

    template <class Meta>
    inline const Real&
    Face<Meta>::getArea() const noexcept
    {
      return (m_area);
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
    {}

    template <class Meta>
    inline Real
    Face<Meta>::unsignedDistance2(const Vec3& a_x0) const noexcept
    {}

    template <class Meta>
    inline Vec3
    Face<Meta>::getSmallestCoordinate() const noexcept
    {}

    template <class Meta>
    inline Vec3
    Face<Meta>::getHighestCoordinate() const noexcept
    {}

    template <class Meta>
    inline void
    Face<Meta>::computeCentroid() noexcept
    {}

    template <class Meta>
    inline void
    Face<Meta>::computeNormal() noexcept
    {}

    template <class Meta>
    inline void
    Face<Meta>::computePolygon2D() noexcept
    {}

    template <class Meta>
    inline void
    Face<Meta>::normalizeNormalVector() noexcept
    {}

    template <class Meta>
    inline void
    Face<Meta>::computeArea() noexcept
    {}

    template <class Meta>
    inline Vec3*
    Face<Meta>::getAllVertexCoordinates() const noexcept
    {}

    template <class Meta>
    inline Vec3
    Face<Meta>::projectPointIntoFacePlane(const Vec3& a_p) const noexcept
    {}

    template <class Meta>
    inline bool
    Face<Meta>::isPointInsideFace(const Vec3& a_p) const noexcept
    {}
  } // namespace DCEL
} // namespace EBGeometry

#endif
