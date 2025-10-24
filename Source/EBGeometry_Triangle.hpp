// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   EBGeometry_Triangle.hpp
 * @brief  Declaration of a triangle struct with signed distance functionality.
 * @author Robert Marskar
 */

#ifndef EBGEOMETRY_TRIANGLE_HPP
#define EBGEOMETRY_TRIANGLE_HPP

// Our includes
#include "EBGeometry_Alignas.hpp"
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_GPUTypes.hpp"
#include "EBGeometry_Macros.hpp"
#include "EBGeometry_Vec.hpp"

namespace EBGeometry {

  /**
   * @brief Triangle struct with signed distance functionality.
   * @details This struct represents a planar triangle and has a signed distance functionality. It is
   * self-contained such that it can be directly copied to GPUs. The struct contains a triangle face normal
   * vector; three vertex positions, and normal vectors for the three vertices and three edges.
   *
   * This struct assumes that the vertices are organized with the right-hand rule. I.e., edges are enumerated
   * as follows:
   *
   * Edge 1 points from vertex 1 to vertex 2
   * Edge 2 points from vertex 2 to vertex 3
   * Edge 3 points from vertex 3 to vertex 0
   *
   * This struct can compute its own normal vector from the vertex positions, and the triangle orientation
   * is then implicitly given by the vertex order.
   *
   * To compute the distance from a point to the triangle, one must determine if the point projects to the
   * "inside" or "outside" of the triangle. This struct contains a 2D embedding of the triangle that can perform
   * this project. If the query point projects to the inside of the triangle, the distance is simply the
   * projected distance onto the triangle plane. If it projects to the outside of the triangle, we check the
   * distance against the triangle edges and vertices.
   */
  template <typename MetaData>
  struct alignas(EBGEOMETRY_ALIGNAS) Triangle
  {
  public:
    /**
     * @brief Default constructor. Does not put the triangle in a usable state.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr Triangle() noexcept = default;

    /**
     * @brief Copy constructor.
     * @param[in] a_otherTriangle Other triangle.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr Triangle(const Triangle& a_otherTriangle) noexcept = default;

    /**
     * @brief Move constructor.
     * @param[in, out] a_otherTriangle Other triangle.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr Triangle(Triangle&& a_otherTriangle) noexcept = default;

    /**
     * @brief Full constructor.
     * @param[in] a_x1 Position of first vertex.
     * @param[in] a_x2 Position of second vertex.
     * @param[in] a_x3 Position of third vertex.     
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr explicit Triangle(const Vec3& a_x1, const Vec3& a_x2, const Vec3& a_x3) noexcept;

    /**
     * @brief Delete constructor to prevent misuse constructor.
     * @param[in] a_vertexPositions Triangle vertex positions
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr Triangle(const Vec3* a_vertexPositions) = delete;

    /**
     * @brief Delete constructor to prevent misuse constructor.
     * @param[in] a_vertexPositions Triangle vertex positions
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr Triangle(std::initializer_list<Vec3> a_vertexPositions) = delete;

    /**
     * @brief Destructor (does nothing).
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr ~Triangle() noexcept = default;

    /**
     * @brief Copy assignment.
     * @param[in] a_otherTriangle Other triangle.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr Triangle&
    operator=(const Triangle& a_otherTriangle) noexcept = default;

    /**
     * @brief Move assignment.
     * @param[in, out] a_otherTriangle Other triangle.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr Triangle&
    operator=(Triangle&& a_otherTriangle) noexcept = default;

    /**
     * @brief Set the triangle normal vector.
     * @param[in] a_normal Normal vector (should be consistent with the vertex ordering!).
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr void
    setNormal(const Vec3& a_normal) noexcept;

    /**
     * @brief Set the triangle vertex positions
     * @param[in] a_x1 Position of first vertex.
     * @param[in] a_x2 Position of second vertex.
     * @param[in] a_x3 Position of third vertex.          
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr void
    setVertexPositions(const Vec3& a_x1, const Vec3& a_x2, const Vec3& a_x3) noexcept;

    /**
     * @brief Set the triangle vertex normals
     * @param[in] a_n1 Normal of first vertex.
     * @param[in] a_n2 Normal of second vertex.
     * @param[in] a_n3 Normal of third vertex.          
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr void
    setVertexNormals(const Vec3& a_n1, const Vec3& a_n2, const Vec3& a_n3) noexcept;

    /**
     * @brief Set the triangle edge normals
     * @param[in] a_n1 Normal of first edge (pointing from first vertex to second vertex)
     * @param[in] a_n2 Normal of second edge (pointing from second vertex to third vertex)
     * @param[in] a_n3 Normal of third edge (pointing from third vertex to first vertex)
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr void
    setEdgeNormals(const Vec3& a_n1, const Vec3& a_n2, const Vec3& a_n3) noexcept;

    /**
     * @brief Set the triangle meta-data
     * @param[in] a_metaData Triangle metadata.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr void
    setMetaData(const MetaData& a_metaData) noexcept;

    /**
     * @brief Compute the triangle normal vector.
     * @details This computes the normal vector from two of the triangle edges.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr void
    computeNormal() noexcept;

    /**
     * @brief Get the triangle normal vector.
     * @return m_triangleNormal
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr Vec3&
    getNormal() noexcept;

    /**
     * @brief Get the triangle normal vector.
     * @return m_triangleNormal
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr const Vec3&
    getNormal() const noexcept;

    /**
     * @brief Get the vertex positions
     * @return m_vertexPositions
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr Vec3*
    getVertexPositions() noexcept;

    /**
     * @brief Get the vertex positions
     * @return m_vertexPositions
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr const Vec3*
    getVertexPositions() const noexcept;

    /**
     * @brief Get the vertex normals
     * @return m_vertexNormals
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr Vec3*
    getVertexNormals() noexcept;

    /**
     * @brief Get the vertex normals
     * @return m_vertexNormals
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr const Vec3*
    getVertexNormals() const noexcept;

    /**
     * @brief Get the edge normals
     * @return m_edgeNormals
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr Vec3*
    getEdgeNormals() noexcept;

    /**
     * @brief Get the edge normals
     * @return m_edgeNormals
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr const Vec3*
    getEdgeNormals() const noexcept;

    /**
     * @brief Get the triangle meta-data
     * @return m_metaData
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr MetaData&
    getMetaData() noexcept;

    /**
     * @brief Get the triangle meta-data
     * @return m_metaData
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr const MetaData&
    getMetaData() const noexcept;

    /**
     * @brief Check if a line passes through the triangle.
     * @details Returns true if the line segment passes through the triangle, edges of the triangle,
     * or through one of the vertices.
     * @param[in] a_x0 One endpoint of the line
     * @param[in] a_x1 Other endpoint of the line
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr bool
    intersects(const Vec3& a_x0, const Vec3& a_x1) const noexcept;

    /**
     * @brief Compute the signed distance from the input point x to the triangle
     * @param[in] a_point Point
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr Real
    value(const Vec3& a_point) const noexcept;

  protected:
    /**
     * @brief Triangle face normal
     */
    Vec3 m_triangleNormal = Vec3::max();

    /**
     * @brief Triangle vertex positions
     */
    Vec3 m_vertexPositions[3]{Vec3::max(), Vec3::max(), Vec3::max()};

    /**
     * @brief Triangle vertex normals
     */
    Vec3 m_vertexNormals[3]{Vec3::max(), Vec3::max(), Vec3::max()};

    /**
     * @brief Triangle edge normals
     */
    Vec3 m_edgeNormals[3]{Vec3::max(), Vec3::max(), Vec3::max()};

    /**
     * @brief Triangle meta-data
     */
    MetaData m_metaData;
  };

  /**
    @brief Simple POD struct that holds squared distance and sign
  */
  struct alignas(EBGEOMETRY_ALIGNAS) DistanceCandidate
  {
    constexpr DistanceCandidate()
    {
      this->m_dist2 = EBGeometry::Limits::max();
      this->m_sgn   = 1;
    }
    
    /**
       @brief Squared absolute distance
    */
    Real m_dist2;

    /**
       @brief Final signed distance = m_abs2 * m_sgn
    */
    int m_sgn;
  };

  /**
    @brief Helper function used when updating the distance to a triangle. Updates the DistanceCandidate if the
    query distance is shorter (absolute value)
    @param[in, out] a_ret Best candidate. 
    @param[in] a_curAbs Candidate distance.
    @param[in] a_retSgn Candidate distance sign
    @param[in] a_mask For turning on/off the distance test. 
  */
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  static void
  compareDistanceHelper(DistanceCandidate& a_ret, Real a_curAbs, int a_curSgn, bool a_mask) noexcept;

  /**
     * @brief Compute squared distance and sign to a single triangle given its SoA fields. 
     * @param[in] a_triangleNormal Face normal of triangle.
     * @param[in] a_vertexPositions Array of vertex positions (length 3).
     * @param[in] a_vertexNormals Array of vertex normals (length 3).
     * @param[in] a_edgeNormals Array of edge normals (length 3).
     * @param[in] a_point Query point.
     * @return Squared distance to triangle and the corresponding sign.
     */
  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  static DistanceCandidate
  signedSquaredDistanceTriangle(const Vec3&                     a_triangleNormal,
                                const Vec3* EBGEOMETRY_RESTRICT a_vertexPositions,
                                const Vec3* EBGEOMETRY_RESTRICT a_vertexNormals,
                                const Vec3* EBGEOMETRY_RESTRICT a_edgeNormals,
                                const Vec3&                     a_point) noexcept;
} // namespace EBGeometry

#include "EBGeometry_TriangleImplem.hpp"

#endif
