/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_Triangle.hpp
  @brief  Declaration of a triangle class with signed distance functionality.
  @author Robert Marskar
*/

#ifndef EBGeometry_Triangle
#define EBGeometry_Triangle

// Our includes
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_GPUTypes.hpp"
#if 1
#warning "Must write the Triangle2D class"
#else
#include "EBGeometry_Triangle2D.hpp"
#endif
#include "EBGeometry_Vec.hpp"

namespace EBGeometry {

  /*!
    @brief Triangle class with signed distance functionality.
    @details This class represents a planar triangle and has a signed distance functionality. It is
    self-contained such that it can be directly copied to GPUs. The class contains a triangle face normal
    vector; three vertex positions, and normal vectors for the three vertices and three edges.

    This class assumes that the vertices are organized with the right-hand rule. I.e., edges are enumerated
    as follows:

    Edge 1 points from vertex 1 to vertex 2
    Edge 2 points from vertex 2 to vertex 3
    Edge 3 points from vertex 3 to vertex 0       

    This class can compute its own normal vector from the vertex positions, and the triangle orientation
    is then implicitly given by the vertex order.

    To compute the distance from a point to the triangle, one must determine if the point projects to the
    "inside" or "outside" of the triangle. This class contains a 2D embedding of the triangle that can perform
    this project. If the query point projects to the inside of the triangle, the distance is simply the
    projected distance onto the triangle plane. If it projects to the outside of the triangle, we check the
    distance against the triangle edges and vertices. 
    the ed
  */
  template <typename MetaData>
  class Triangle
  {
  public:
    /*!
      @brief Default constructor. Does not put the triangle in a usable state.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline Triangle() noexcept;

    /*!
      @brief Copy constructor. Copies all data members from the other triangle.
      @param[in] a_otherTriangle Other triangle.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline Triangle(const Triangle& a_otherTriangle) noexcept;

    /*!
      @brief Destructor (does nothing).
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline ~Triangle() noexcept;

    /*!
      @brief Set the triangle normal vector.
      @param[in] a_normal Normal vector (should be consistent with the vertex ordering!).
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline void
    setNormal(const Vec3& a_normal) noexcept;

    /*!
      @brief Set the triangle vertex positions
      @param[in] a_vertexPositions Vertex positions
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline void
    setVertexPositions(const Vec3 a_vertexPositions[3]) noexcept;

    /*!
      @brief Set the triangle vertex normals
      @param[in] a_vertexNormals Vertex normals
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline void
    setVertexNormals(const Vec3 a_vertexNormals[3]) noexcept;

    /*!
      @brief Set the triangle edge normals
      @param[in] a_edgeNormals Edge normals
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline void
    setEdgeNormals(const Vec3 a_edgeNormals[3]) noexcept;

    /*!
      @brief Set the triangle meta-data
      @param[in] a_metaData Triangle metadata.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline void
    setMetaData(const MetaData& a_metaData) noexcept;

    /*!
      @brief Compute the triangle normal vector.
      @details This computes the normal vector from two of the triangle edges.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline void
    computeNormal() noexcept;

    /*!
      @brief Get the triangle normal vector.
      @return m_normal
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Vec3&
    getNormal() noexcept;

    /*!
      @brief Get the triangle normal vector.
      @return m_normal
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline const Vec3&
    getNormal() const noexcept;

    /*!
      @brief Get the vertex positions
      @return m_vertexPositions
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Vec3*
    getVertexPositions() noexcept;

    /*!
      @brief Get the vertex positions
      @return m_vertexPositions
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline const Vec3*
    getVertexPositions() const noexcept;

    /*!
      @brief Get the vertex normals
      @return m_vertexNormals
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Vec3*
    getVertexNormals() noexcept;

    /*!
      @brief Get the vertex normals
      @return m_vertexNormals
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline const Vec3*
    getVertexNormals() const noexcept;

    /*!
      @brief Get the edge normals
      @return m_edgeNormals
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Vec3*
    getEdgeNormals() noexcept;

    /*!
      @brief Get the edge normals
      @return m_edgeNormals
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline const Vec3*
    getEdgeNormals() const noexcept;

    /*!
      @brief Get the triangle meta-data
      @return m_metaData
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline MetaData&
    getMetaData() noexcept;

    /*!
      @brief Get the triangle meta-data
      @return m_metaData
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline const MetaData&
    getMetaData() const noexcept;

    /*!
      @brief Compute the signed distance to the triangle
      @param[in] a_x Point
      @return Returns the shorter signed distance from a_x to the triangle. 
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Real
    signedDistance(const Vec3& a_x) noexcept;

  protected:
    /*!
      @brief Triangle face normal
    */
    Vec3 m_normal;

    /*!
      @brief Triangle vertex positions
    */
    Vec3 m_vertexPositions[3];

    /*!
      @brief Triangle vertex normals
    */
    Vec3 m_vertexNormals[3];
    /*!
      @brief Triangle edge normals
    */
    Vec3 m_edgeNormals[3];

    /*!
      @brief Triangle meta-data normals
    */
    MetaData m_metaData;

#if 1
#warning "Must write the Triangle2D class"
#else
    /*!
      @brief 2D projection of the triangle to one of the Cartesian coordinate directions
    */
    Triangle2D m_triangle2D;
#endif
  };
} // namespace EBGeometry

#include "EBGeometry_TriangleImplem.hpp"

#endif