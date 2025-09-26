/**
 * @file    EBGeometry_TriangleMeshImplem.hpp
 * @brief   Implementation of EBGeometry_TriangleMesh.hpp
 * @author  Robert Marskar
 */

#ifndef EBGeometry_TriangleMeshImplem
#define EBGeometry_TriangleMeshImplem

// Our includes
#include "EBGeometry_TriangleMesh.hpp"
#include "EBGeometry_Macros.hpp"

namespace EBGeometry {

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  TriangleMesh<MetaData>::TriangleMesh(long long int a_numTriangles, const Triangle<MetaData>* a_triangles) noexcept
  {
    EBGEOMETRY_ALWAYS_EXPECT(a_numTriangles >= 1);
    EBGEOMETRY_ALWAYS_EXPECT(a_triangles != nullptr);

    this->define(a_numTriangles, a_triangles);
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  void
  TriangleMesh<MetaData>::define(long long int a_numTriangles, const Triangle<MetaData>* a_triangles) noexcept
  {
    EBGEOMETRY_ALWAYS_EXPECT(a_numTriangles >= 1);
    EBGEOMETRY_ALWAYS_EXPECT(a_triangles != nullptr);

    m_numTriangles = a_numTriangles;
    m_triangles    = a_triangles;
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  long long int
  TriangleMesh<MetaData>::getNumberOfTriangles() const noexcept
  {
    return m_numTriangles;
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  const Triangle<MetaData>*
  TriangleMesh<MetaData>::getTriangles() const noexcept
  {
    return m_triangles;
  }

  template <class MetaData>
  EBGEOMETRY_ALWAYS_INLINE
  Real
  TriangleMesh<MetaData>::value(const Vec3& a_point) const noexcept
  {
    EBGEOMETRY_EXPECT(m_numTriangles >= 1);
    EBGEOMETRY_EXPECT(m_triangles != nullptr);

    Real dist = EBGeometry::Limits::max();

    for (long long int i = 0; i < this->m_numTriangles; i++) {
      const Real curDist = m_triangles[i].signedDistance(a_point);

      dist = (abs(curDist) < abs(dist)) ? curDist : dist;
    }

    return dist;
  }
} // namespace EBGeometry

#endif
