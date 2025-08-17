/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file    EBGeometry_AnalyticDistanceFunctions.hpp
  @brief   Declaration of various analytic distance functions.
  @author  Robert Marskar
*/

#ifndef EBGeometry_AnalyticDistanceFunctions
#define EBGeometry_AnalyticDistanceFunctions

#include "EBGeometry_Types.hpp"
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_GPUTypes.hpp"
#include "EBGeometry_ImplicitFunction.hpp"
#include "EBGeometry_Macros.hpp"
#include "EBGeometry_Vec.hpp"

namespace EBGeometry {

  /*!
    @brief Signed distance function for a plane.
    @details User species a point on the plane and the outward normal vector. 
  */
  class PlaneSDF : public ImplicitFunction
  {
  public:
    /*!
      @brief Default constructor.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    PlaneSDF() noexcept = default;

    /*!
      @brief Full constructor
      @param[in] a_point      Point on the plane
      @param[in] a_normal     Plane normal vector.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    PlaneSDF(const Vec3& a_point, const Vec3& a_normal) noexcept :
      m_point(a_point),
      m_normal(a_normal)
    {
      EBGEOMETRY_EXPECT(m_normal.length() > EBGeometry::Limits::min());

      m_normal = m_normal / m_normal.length();
    }

    /*!
      @brief Copy constructor. 
      @param[in] a_plane Other plane
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    PlaneSDF(const PlaneSDF& a_plane) noexcept = default;

    /*!
      @brief Move constructor. 
      @param[in, out] a_plane Other plane
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    PlaneSDF(PlaneSDF&& a_plane) noexcept = default;

    /*!
      @brief Destructor
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    ~PlaneSDF() noexcept override = default;

    /*!
      @brief Copy assignmnt. 
      @param[in] a_plane Other plane
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    PlaneSDF&
    operator=(const PlaneSDF& a_plane) noexcept = default;

    /*!
      @brief Move assignment. 
      @param[in, out] a_plane Other plane
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    PlaneSDF&
    operator=(PlaneSDF&& a_plane) noexcept = default;

    /*!
      @brief Signed distance function for sphere.
      @param[in] a_point Position.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Real
    value(const Vec3& a_point) const noexcept override
    {
      return dot((a_point - m_point), m_normal);
    }

  protected:
    /*!
      @brief Point on plane
    */
    Vec3 m_point = Vec3::zero();

    /*!
      @brief Plane normal vector
    */
    Vec3 m_normal = Vec3::one();
  };

  /*!
    @brief Signed distance field for sphere.
    @details User specifies the center and radius. 
  */
  class SphereSDF : public ImplicitFunction
  {
  public:
    /*!
      @brief Default constructor. Sets the unit sphere. 
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    SphereSDF() noexcept = default;

    /*!
      @brief Full constructor.
      @param[in] a_center Sphere center
      @param[in] a_radius Sphere radius
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    SphereSDF(const Vec3& a_center, const Real& a_radius) noexcept :
      m_center(a_center),
      m_radius(a_radius)
    {
      EBGEOMETRY_EXPECT(m_radius > 0.0);
    }

    /*!
      @brief Copy constructor.
      @param[in] a_sphere Other sphere
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    SphereSDF(const SphereSDF& a_sphere) noexcept = default;

    /*!
      @brief Move constructor.
      @param[in, out] a_sphere Other sphere
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    SphereSDF(SphereSDF&& a_sphere) noexcept = default;

    /*!
      @brief Destructor.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    ~SphereSDF() noexcept override = default;

    /*!
      @brief Copy assignment.
      @param[in] a_sphere Other sphere. 
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    SphereSDF&
    operator=(const SphereSDF& a_sphere) noexcept = default;

    /*!
      @brief Move assignment.
      @param[in, out] a_sphere Other sphere. 
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    SphereSDF&
    operator=(SphereSDF&& a_sphere) noexcept = default;

    /*!
      @brief Signed distance function for sphere.
      @param[in] a_point Position.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Real
    value(const Vec3& a_point) const noexcept override
    {
      return (a_point - m_center).length() - m_radius;
    }

  protected:
    /*!
      @brief Sphere center
    */
    Vec3 m_center = Vec3::zero();

    /*!
      @brief Sphere radius
    */
    Real m_radius = 1.0;
  };

  /*!
    @brief Signed distance field for an axis-aligned box.
    @details User inputs low and high corners of the box. 
  */
  class BoxSDF : public ImplicitFunction
  {
  public:
    /*!
      @brief Default constructor. Sets the unit box. 
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    BoxSDF() noexcept = default;

    /*!
      @brief Full constructor. Sets the low and high corner. 
      @details One must always have m_loCorner < m_hiCorner for all coordinate directions.
      @param[in] a_loCorner   Lower left corner
      @param[in] a_hiCorner   Upper right corner
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    BoxSDF(const Vec3& a_loCorner, const Vec3& a_hiCorner) noexcept :
      m_loCorner(a_loCorner),
      m_hiCorner(a_hiCorner)
    {
      EBGEOMETRY_EXPECT(m_loCorner[0] < m_hiCorner[0]);
      EBGEOMETRY_EXPECT(m_loCorner[1] < m_hiCorner[1]);
      EBGEOMETRY_EXPECT(m_loCorner[2] < m_hiCorner[2]);
    }

    /*!
      @brief Copy constructor. 
      @param[in] a_box Other box
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    BoxSDF(const BoxSDF& a_box) noexcept = default;

    /*!
      @brief Move constructor. 
      @param[in, out] a_box Other box
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    BoxSDF(BoxSDF&& a_box) noexcept = default;

    /*!
      @brief Destructor
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    ~BoxSDF() noexcept override = default;

    /*!
      @brief Copy assignment operator
      @param[in] a_box Other box
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    BoxSDF&
    operator=(const BoxSDF& a_box) noexcept = default;

    /*!
      @brief Move assignment operator
      @param[in, out] a_box Other box
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    BoxSDF&
    operator=(BoxSDF&& a_box) noexcept = default;

    /*!
      @brief Signed distance function for a box.
      @param[in] a_point Position.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Real
    value(const Vec3& a_point) const noexcept override
    {
      // For each coordinate direction, we have delta[dir] if a_point[dir] falls  \
      // between xLo and xHi. In this case delta[dir] will be the signed distance \
      // to the closest box face in the dir-direction. Otherwise, if a_point[dir] \
      // is outside the corner we have delta[dir] > 0.
      const Vec3 delta(EBGeometry::max(m_loCorner[0] - a_point[0], a_point[0] - m_hiCorner[0]),
                       EBGeometry::max(m_loCorner[1] - a_point[1], a_point[1] - m_hiCorner[1]),
                       EBGeometry::max(m_loCorner[2] - a_point[2], a_point[2] - m_hiCorner[2]));

      // Note: max is max(Vec3, Vec3) and not EBGeometry::max. It returns a
      // vector with coordinate-wise largest components. Note that the first part
      // EBGeometry::min(...) is the signed distance on the inside of the box (delta will
      // have negative components). The other part max(Vec3::zero(), ...) is
      // for outside the box.
      const Real d = EBGeometry::min(Real(0.0), delta[delta.maxDir(false)]) + max(Vec3::zero(), delta).length();

      return d;
    }

  protected:
    /*!
      @brief Low box corner
    */
    Vec3 m_loCorner = -0.5 * Vec3::one();

    /*!
      @brief High box corner
    */
    Vec3 m_hiCorner = 0.5 * Vec3::one();
  };

  /*!
    @brief Signed distance field for a torus
    @details User inputs the center, major radius, and minor radius. The torus always lies in the xy plane
  */
  class TorusSDF : public ImplicitFunction
  {
  public:
    /*!
      @brief Default constructor. Sets a torus in the origin with a major radius 1 and minor radius 0.1
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    TorusSDF() noexcept = default;

    /*!
      @brief Full constructor. Sets the center and the radii. The torus lies in the xy plane.
      @details One must always have minor radius < major_radius
      @param[in] a_center Torus center.
      @param[in] a_majorRadius Torus major radius.
      @param[in] a_minorRadius Torus minor radius.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    TorusSDF(const Vec3& a_center, const Real& a_majorRadius, const Real& a_minorRadius) noexcept :
      m_center(a_center),
      m_majorRadius(a_majorRadius),
      m_minorRadius(a_minorRadius)
    {
      EBGEOMETRY_EXPECT(m_minorRadius < m_majorRadius);
    }

    /*!
      @brief Copy constructor. 
      @param[in] a_torus Other torus
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    TorusSDF(const TorusSDF& a_torus) noexcept = default;

    /*!
      @brief Move constructor. 
      @param[in, out] a_torus Other torus
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    TorusSDF(TorusSDF&& a_torus) noexcept = default;

    /*!
      @brief Destructor
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    ~TorusSDF() noexcept override = default;

    /*!
      @brief Copy assignment operator
      @param[in] a_torus Other torus
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    TorusSDF&
    operator=(const TorusSDF& a_torus) noexcept = default;

    /*!
      @brief Move assignment operator
      @param[in, out] a_torus Other torus
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    TorusSDF&
    operator=(TorusSDF&& a_torus) noexcept = default;

    /*!
      @brief Signed distance function for the torus.
      @param[in] a_point Position.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Real
    value(const Vec3& a_point) const noexcept override
    {
      const auto p   = a_point - m_center;
      const auto rho = sqrt(p[0] * p[0] + p[1] * p[1]) - m_majorRadius;
      const auto d   = sqrt(rho * rho + p[2] * p[2]) - m_minorRadius;

      return Real(d);
    }

  protected:
    /*!
      @brief Torus center.
    */
    Vec3 m_center = Vec3::zero();

    /*!
      @brief Torus major radius
    */
    Real m_majorRadius;

    /*!
      @brief Torus minor radius
    */
    Real m_minorRadius;
  };
} // namespace EBGeometry

#endif
