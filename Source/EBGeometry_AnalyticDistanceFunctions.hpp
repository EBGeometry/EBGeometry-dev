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
    [[nodiscard]] inline Real
    operator()(const Vec3& a_point) const noexcept
    {
      return dot((a_point - m_point), m_normal);
    }

    /*!
      @brief Copy plane to the GPU.
      @details This allocates memory and performs the copy. It is up to the user to free
      the memory if it is no longer needed by using the freeFromGPU() call.
    */
    EBGEOMETRY_GPU_HOST
    inline PlaneSDF*
    putOnGPU() const noexcept
    {
      PlaneSDF* plane;

      cudaMalloc((void**)&plane, sizeof(PlaneSDF));

      cudaMemcpy(plane, &(*this), sizeof(PlaneSDF), cudaMemcpyHostToDevice);

      return plane;
    }

    /*!
      @brief Free memory from the GPU
      @details Free memory from GPU if the plane is no longer on the device.
    */
    EBGEOMETRY_GPU_HOST
    inline void
    freeFromGPU() noexcept
    {
      cudaFree(&(*this));
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

  /*!
    @brief Signed distance field for sphere.
    @details User specifies the center and radius. 
  */
  class SphereSDF
  {
  public:
    /*!
      @brief Default constructor. Sets center ot zero and radius to one.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline SphereSDF() noexcept
    {
      m_center = Vec3::zero();
      m_radius = 1.0;
    }

    /*!
      @brief Full constructor.
      @param[in] a_center Sphere center
      @param[in] a_radius Sphere radius
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline SphereSDF(const Vec3& a_center, const Real& a_radius) noexcept
    {
      this->m_center = a_center;
      this->m_radius = a_radius;
    }

    /*!
      @brief Signed distance function for sphere.
      @param[in] a_point Position.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline Real
    operator()(const Vec3& a_point) const noexcept
    {
      return (a_point - m_center).length() - m_radius;
    }

  protected:
    /*!
      @brief Sphere center
    */
    Vec3 m_center;

    /*!
      @brief Sphere radius
    */
    Real m_radius;
  };
} // namespace EBGeometry

static_assert(EBGeometry::ImplicitFunction<EBGeometry::PlaneSDF>);
static_assert(EBGeometry::ImplicitFunction<EBGeometry::SphereSDF>);

#endif
