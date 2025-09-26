/**
 * @file   EBGeometry_DCEL_Polygon2D.hpp
 * @brief  Declaration of a two-dimensional polygon class for embedding 3D
 * polygon faces
 * @author Robert Marskar
 */

#ifndef EBGeometry_Polygon2D
#define EBGeometry_Polygon2D

// Our includes
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_GPUTypes.hpp"
#include "EBGeometry_Macros.hpp"
#include "EBGeometry_Vec.hpp"

namespace EBGeometry::DCEL {
  /**
   * @brief Class for embedding a polygon face into 2D.
   * @details This class is required for determining whether or not a 3D point
   * projected to the plane of an N-sided polygon lies inside or outside the
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
  class Polygon2D
  {
  public:
    /**
     * @brief Supported algorithms for performing inside/outside tests when
     * checking if a point projects to the inside or outside of a polygon face.
     */
    enum class InsideOutsideAlgorithm // NOLINT (clang-tidy might complain that size is too large)
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
    Polygon2D() noexcept = default;

    /**
     * @brief Full constructor
     * @param[in] a_normal Normal vector of the 3D polygon face
     * @param[in] a_numPoints Number of vertices
     * @param[in] a_points Vertex coordinates of the 3D polygon face
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Polygon2D(const Vec3& a_normal, int a_numPoints, const Vec3* a_points) noexcept;

    /**
     * @brief Copy constructor.
     * @param[in] a_polygon2D Object to copy
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Polygon2D(const Polygon2D& a_polygon2D) noexcept;

    /**
     * @brief Move constructor (but performs a copy anyways).
     * @param[in, out] a_polygon2D Object to move
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Polygon2D(Polygon2D&& a_polygon2D) noexcept;

    /**
     * @brief Destructor (does nothing)
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    ~Polygon2D() noexcept;

    /**
     * @brief Copy assignment.
     * @param[in] a_polygon2D Object to copy
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Polygon2D&
    operator=(const Polygon2D& a_polygon2D) noexcept;

    /**
     * @brief Move assignment (but performs a copy anyways)
     * @param[in, out] a_polygon2D Object to move
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Polygon2D&
    operator=(Polygon2D&& a_polygon2D) noexcept;

    /**
     * @brief Define function. This find the direction to ignore and then computes
     * the 2D points.
     * @param[in] a_normal Normal vector for polygon face
     * @param[in] a_numPoints Number of vertices
     * @param[in] a_points Vertex coordinates for polygon face.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    define(const Vec3& a_normal, int a_numPoints, const Vec3* a_points) noexcept;

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
     * @brief Number of vertex points.
     */
    int m_numPoints = -1;

    /**
     * @brief List of points in 2D.
     * @details This is the position of the vertices, projected into 2D
     */
    Vec2* m_points = nullptr;

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

#include "EBGeometry_DCEL_Polygon2DImplem.hpp" // NOLINT

#endif
