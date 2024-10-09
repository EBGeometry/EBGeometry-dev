/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_TriangleImplem.hpp
  @brief  Implementation of EBGeometry_Triangle.hpp
  @author Robert Marskar
*/

#ifndef EBGeometry_TriangleImplem
#define EBGeometry_TriangleImplem

// Our includes
#include "EBGeometry_Triangle.hpp"

namespace EBGeometry {

  template <typename MetaData>
  inline Triangle<MetaData>::Triangle() noexcept
  {
    this->m_normal = Vec3::zero();

    for (int i = 0; i < 3; i++) {
      this->m_vertexPositions[i] = Vec3::zero();
      this->m_vertexNormals[i]   = Vec3::zero();
      this->m_edgeNormals[i]     = Vec3::zero();
    }
  }

  template <typename MetaData>
  inline Triangle<MetaData>::Triangle(const Triangle& a_otherTriangle) noexcept
  {
    this->m_normal = a_otherTriangle.m_normal;

    for (int i = 0; i < 3; i++) {
      this->m_vertexPositions[i] = a_otherTriangle.m_vertexPositions[i];
      this->m_vertexNormals[i]   = a_otherTriangle.m_vertexNormals[i];
      this->m_edgeNormals[i]     = a_otherTriangle.m_edgeNormals[i];
    }

    this->m_metaData   = a_otherTriangle.m_metaData;
    this->m_triangle2D = a_otherTriangle.m_triangle2D;
  }

  template <typename MetaData>
  inline Triangle<MetaData>::~Triangle() noexcept
  {}

  template <typename MetaData>
  inline void
  Triangle<MetaData>::setNormal(const Vec3& a_normal) noexcept
  {
    this->m_normal = a_normal;
  }

  template <typename MetaData>
  inline void
  Triangle<MetaData>::setVertexPositions(const Vec3 a_vertexPositions[3]) noexcept
  {
    for (int i = 0; i < 3; i++) {
      this->m_vertexPositions[i] = a_vertexPositions[i];
    }
  }

  template <typename MetaData>
  inline void
  Triangle<MetaData>::setVertexNormals(const Vec3 a_vertexNormals[3]) noexcept
  {
    for (int i = 0; i < 3; i++) {
      this->m_vertexNormals[i] = a_vertexNormals[i];
    }
  }

  template <typename MetaData>
  inline void
  Triangle<MetaData>::setEdgeNormals(const Vec3 a_edgeNormals[3]) noexcept
  {
    for (int i = 0; i < 3; i++) {
      this->m_edgeNormals[i] = a_edgeNormals[i];
    }
  }

  template <typename MetaData>
  inline void
  Triangle<MetaData>::setMetaData(const MetaData& a_metaData) noexcept
  {
    this->m_metaData = a_metaData;
  }

  template <typename MetaData>
  inline void
  Triangle<MetaData>::computeNormal() noexcept
  {
#warning "Triangle::computeNormal -- not implemented";
  }

  template <typename MetaData>
  [[nodiscard]] inline Vec3&
  Triangle<MetaData>::getNormal() noexcept
  {
    return (this->m_normal);
  }

  template <typename MetaData>
  [[nodiscard]] inline const Vec3&
  Triangle<MetaData>::getNormal() const noexcept
  {
    return (this->m_normal);
  }

  template <typename MetaData>
  [[nodiscard]] inline Vec3*
  Triangle<MetaData>::getVertexPositions() noexcept
  {
    return (this->m_vertexPositions);
  }

  template <typename MetaData>
  [[nodiscard]] inline const Vec3*
  Triangle<MetaData>::getVertexPositions() const noexcept
  {
    return (this->m_vertexPositions);
  }

  template <typename MetaData>
  [[nodiscard]] inline Vec3*
  Triangle<MetaData>::getVertexNormals() noexcept
  {
    return (this->m_vertexNormals);
  }

  template <typename MetaData>
  [[nodiscard]] inline const Vec3*
  Triangle<MetaData>::getVertexNormals() const noexcept
  {
    return (this->m_vertexNormals);
  }
  template <typename MetaData>
  [[nodiscard]] inline Vec3*
  Triangle<MetaData>::getEdgeNormals() noexcept
  {
    return (this->m_edgeNormals);
  }

  template <typename MetaData>
  [[nodiscard]] inline const Vec3*
  Triangle<MetaData>::getEdgeNormals() const noexcept
  {
    return (this->m_edgeNormals);
  }

  template <typename MetaData>
  [[nodiscard]] inline MetaData&
  Triangle<MetaData>::getMetaData() noexcept
  {
    return (this->m_metaData);
  }

  template <typename MetaData>
  [[nodiscard]] inline const MetaData&
  Triangle<MetaData>::getMetaData() const noexcept
  {
    return (this->m_metaData);
  }

  template <typename MetaData>
  [[nodiscard]] inline Real
  Triangle<MetaData>::signedDistance(const Vec3& a_x) noexcept
  {
#warning "Triangle::signedDistance -- not implemented"
    return 0.0;
  }
} // namespace EBGeometry

#endif
