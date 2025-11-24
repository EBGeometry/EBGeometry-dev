// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   EBGeometry_DCEL_EmbeddedFace2D.hpp
 * @brief  Declaration of a two-dimensional polygon class for embedding 3D
 * polygon faces
 * @author Robert Marskar
 */

#ifndef EBGEOMETRY_DCEL_EMBEDDEDFACE2D_HPP
#define EBGEOMETRY_DCEL_EMBEDDEDFACE2D_HPP

// Our includes
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_GPUTypes.hpp"
#include "EBGeometry_Macros.hpp"
#include "EBGeometry_Span.hpp"
#include "EBGeometry_Vec.hpp"

namespace EBGeometry::DCEL {
  /**
   * @brief Class for embedding a polygon face into 2D.
   * @details This class is required for determining whether or not a 3D point
   * projected to the plane of an N-sided planar polygon lies inside or outside the
   * polygon face. To do this we compute the 2D embedding of the polygon face,
   * reducing the problem to a tractable dimension where we can use well-tested
   * algorithm. The 2D embedding of a polygon occurs by taking a set of 3D points
   * and a corresponding normal vector, and projecting those points along one of
   * the 3D Cartesian axes such that the polygon has the largest area. In essence,
   * we simply find the direction with the smallest normal vector component and
   * ignore that. Once the 2D embedding is computed, we can use well-known
   * algorithms for checking if a point lies inside or outside. The supported
   * algorithms are 1) The winding number algorithm (computing the winding number),
   * 2) Computing the subtended angle of the point with the edges of the polygon
   * (sums to 360 degrees if the point is inside), or computing the crossing number
   * which checks how many times a ray cast from the point crosses the edges of the
   * polygon.
   */
  class EmbeddedFace2D
  {
  public:
    /**
     * @brief Supported algorithms for performing inside/outside tests when
     * checking if a point projects to the inside or outside of a polygon face.
     */
    enum class InsideOutsideAlgorithm
    {
      SubtendedAngle,
      CrossingNumber,
      WindingNumber
    };

    /**
     * @brief Default constructor.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    EmbeddedFace2D() noexcept = default;

    /**
     * @brief Full constructor
     * @param[in] a_normal Normal vector of the 3D polygon face
     * @param[in] a_points Span view of vertex coordinates of the 3D polygon face
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    EmbeddedFace2D(const Vec3& a_normal, EBGeometry::Span<const Vec3> a_points) noexcept;

    /**
     * @brief Copy constructor.
     * @param[in] a_embeddedFace2D Object to copy
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    EmbeddedFace2D(const EmbeddedFace2D& a_embeddedFace2D) noexcept = default;

    /**
     * @brief Move constructor.
     * @param[in, out] a_embeddedFace2D Object to move
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    EmbeddedFace2D(EmbeddedFace2D&& a_embeddedFace2D) noexcept = default;

    /**
     * @brief Destructor (does nothing)
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    ~EmbeddedFace2D() noexcept = default;

    /**
     * @brief Copy assignment.
     * @param[in] a_embeddedFace2D Object to copy
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    EmbeddedFace2D&
    operator=(const EmbeddedFace2D& a_embeddedFace2D) noexcept = default;

    /**
     * @brief Move assignment.
     * @param[in, out] a_embeddedFace2D Object to move
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    EmbeddedFace2D&
    operator=(EmbeddedFace2D&& a_embeddedFace2D) noexcept = default;

    /**
     * @brief Define function. This finds the direction to ignore and then computes
     * the 2D points.
     * @param[in] a_normal Normal vector for polygon face
     * @param[in] a_points Span view of vertex coordinates for polygon face
     * @param[in] a_polygon2DPoints Span view for storing the computed 2D points
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    define(const Vec3&                  a_normal,
           EBGeometry::Span<const Vec3> a_points,
           EBGeometry::Span<Vec2>       a_polygon2DPoints) noexcept;

    /**
     * @brief Check if a point is inside or outside the 2D polygon
     * @param[in] a_point     3D point coordinates
     * @param[in] a_algorithm Inside/outside algorithm
     * @details This will call the function corresponding to a_algorithm.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    bool
    isPointInside(const Vec3& a_point, InsideOutsideAlgorithm a_algorithm) const noexcept;

    /**
     * @brief Check if a point is inside a 2D polygon, using the winding number
     * algorithm
     * @param[in] a_point 3D point coordinates
     * @return Returns true if the 3D point projects to the inside of the 2D
     * polygon
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    bool
    isPointInsidePolygonWindingNumber(const Vec3& a_point) const noexcept;

    /**
     * @brief Check if a point is inside a 2D polygon, by computing the number of
     * times a ray crosses the polygon edges.
     * @param[in] a_point 3D point coordinates
     * @return Returns true if the 3D point projects to the inside of the 2D
     * polygon
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    bool
    isPointInsidePolygonCrossingNumber(const Vec3& a_point) const noexcept;

    /**
     * @brief Check if a point is inside a 2D polygon, using the subtended angles
     * @param[in] a_point 3D point coordinates
     * @return Returns true if the 3D point projects to the inside of the 2D
     * polygon
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    bool
    isPointInsidePolygonSubtend(const Vec3& a_point) const noexcept;

  protected:
    /**
     * @brief The corresponding 2D x-direction (one direction is ignored)
     */
    int m_xDir = -1;

    /**
     * @brief The corresponding 2D y-direction (one direction is ignored)
     */
    int m_yDir = -1;

    /**
     * @brief Non-owning view of 2D polygon vertex coordinates.
     * @details This is a Span view into a stack-allocated array owned by Face.
     * Contains the position of the vertices, projected into 2D.
     */
    EBGeometry::Span<const Vec2> m_points{};

    /**
     * @brief Project a 3D point onto the 2D polygon plane (this ignores one of the
     * vector components)
     * @param[in] a_point 3D point
     * @return 2D point, ignoring one of the coordinate directions.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Vec2
    projectPoint(const Vec3& a_point) const noexcept;

    /**
     * @brief Compute the winding number for a point P with the 2D polygon
     * @param[in] a_point 2D point
     * @return Returns winding number.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    int
    computeWindingNumber(const Vec2& a_point) const noexcept;

    /**
     * @brief Compute the crossing number for a point P with the 2D polygon
     * @param[in] a_point 2D point
     * @return Returns crossing number.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    int
    computeCrossingNumber(const Vec2& a_point) const noexcept;

    /**
     * @brief Compute the subtended angle for a point a_point with the 2D polygon
     * @param[in] a_point 2D point
     * @return Returns subtended angle.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Real
    computeSubtendedAngle(const Vec2& a_point) const noexcept;
  };
} // namespace EBGeometry::DCEL

#include "EBGeometry_DCEL_EmbeddedFace2DImplem.hpp" // NOLINT

#endif
