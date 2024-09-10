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
#include "EBGeometry_GPUTypes.hpp"
#include "EBGeometry_Vec.hpp"

namespace EBGeometry {

  /*!
    @brief Signed distance function for a plane.
    @details User species a point on the plane and the outward normal vector.
  */
  class PlaneSDF
  {
  public:
    /*!
      @brief Disallowed weak construction
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline PlaneSDF() noexcept
    {
      m_normal = Vec3::one();
      m_point  = Vec3::zero();
    }

    /*!
      @brief Full constructor
      @param[in] a_point      Point on the plane
      @param[in] a_normal     Plane normal vector.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline PlaneSDF(const Vec3& a_point, const Vec3& a_normal) noexcept
    {
      m_point  = a_point;
      m_normal = a_normal / a_normal.length();
    }

    /*!
      @brief Signed distance function for sphere.
      @param[in] a_point Position.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Real operator()(const Vec3& a_point) const noexcept
    {
      return dot((a_point - m_point), m_normal);
    }

  protected:
    /*!
      @brief Point on plane
    */
    Vec3 m_point;

    /*!
      @brief Plane normal vector
    */
    Vec3 m_normal;
  };
} // namespace EBGeometry

static_assert(EBGeometry::ImplicitFunction<EBGeometry::PlaneSDF>);

#endif

