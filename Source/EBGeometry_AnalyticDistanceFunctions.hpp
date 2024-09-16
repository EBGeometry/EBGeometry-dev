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
      @brief Destructor
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    virtual ~PlaneSDF() noexcept
    {}

    /*!
      @brief Signed distance function for sphere.
      @param[in] a_point Position.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Real
    value(const Vec3& a_point) const noexcept override
    {
      return dot((a_point - m_point), m_normal);
    }

    /*!
      @brief Factory method for building the implicit function on the GPU.
    */
#ifdef EBGEOMETRY_ENABLE_GPU
    EBGEOMETRY_GPU_HOST
    [[nodiscard]] inline void*
    putOnGPU() const noexcept override
    {
      GPUPointer<PlaneSDF> plane = allocateImplicitFunctionOnDevice<PlaneSDF>();

      Vec3* point;
      Vec3* normal;

      //      cudaMalloc((void**)&plane, sizeof(GPUPointer<PlaneSDF>));
      cudaMalloc((void**)&point, sizeof(Vec3));
      cudaMalloc((void**)&normal, sizeof(Vec3));

      cudaMemcpy(point, &m_point, sizeof(Vec3), cudaMemcpyHostToDevice);
      cudaMemcpy(normal, &m_normal, sizeof(Vec3), cudaMemcpyHostToDevice);

      createImplicitFunctionOnDevice<<<1, 1>>>(plane, point, normal);

      return plane;
    }
#endif

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
    @brief Factory method for building UnionIF objects on the host or device.
  */
  class PlaneSDFFactory : public ImplicitFunctionFactory<PlaneSDF>
  {
  public:
    EBGEOMETRY_GPU_HOST
    PlaneSDFFactory(const Vec3* a_point, const Vec3* a_normal) noexcept
    {
      m_point = a_point;
      m_normal = a_normal;
    };

    EBGEOMETRY_GPU_HOST
    [[nodiscard]] virtual PlaneSDF*
    buildOnHost() const noexcept override
    {
      return new PlaneSDF(*m_point, *m_normal);
    }

    /*!
      @brief Build implicit function on the device
    */
    EBGEOMETRY_GPU_HOST
    [[nodiscard]] virtual PlaneSDF**
    buildOnDevice() const noexcept override
    {
      PlaneSDF** plane = allocateImplicitFunctionOnDevice<PlaneSDF>();

      createImplicitFunctionOnDevice<<<1, 1>>>(plane, m_point, m_normal);

      return plane;
    };
  protected:

    const Vec3* m_point;

    const Vec3* m_normal;    
  };

  /*!
    @brief Signed distance field for sphere.
    @details User specifies the center and radius. 
  */
  class SphereSDF : public ImplicitFunction
  {
  public:
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
      @brief Destructor
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    virtual ~SphereSDF() noexcept
    {}

    /*!
      @brief Factory method for building the implicit function on the GPU.
    */
#ifdef EBGEOMETRY_ENABLE_GPU
    EBGEOMETRY_GPU_HOST
    [[nodiscard]] inline void*
    putOnGPU() const noexcept override
    {
      GPUPointer<SphereSDF> sphere = allocateImplicitFunctionOnDevice<SphereSDF>();

      Vec3* sphereCenter;
      Real* sphereRadius;

      cudaMalloc((void**)&sphereCenter, sizeof(Vec3));
      cudaMalloc((void**)&sphereRadius, sizeof(Real));

      cudaMemcpy(sphereCenter, &m_center, sizeof(Vec3), cudaMemcpyHostToDevice);
      cudaMemcpy(sphereRadius, &m_radius, sizeof(Real), cudaMemcpyHostToDevice);

      createImplicitFunctionOnDevice<<<1, 1>>>(sphere, sphereCenter, sphereRadius);

      return sphere;
    }
#endif

    /*!
      @brief Signed distance function for sphere.
      @param[in] a_point Position.
    */
    EBGEOMETRY_GPU_HOST_DEVICE [[nodiscard]] inline Real
    value(const Vec3& a_point) const noexcept override
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

  /*!
    @brief Signed distance field for an axis-aligned box.
    @details User inputs low and high corners of the box. 
  */
  class BoxSDF : public ImplicitFunction
  {
  public:
    /*!
      @brief Full constructor. Sets the low and high corner
      @param[in] a_loCorner   Lower left corner
      @param[in] a_hiCorner   Upper right corner
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline BoxSDF(const Vec3& a_loCorner, const Vec3& a_hiCorner) noexcept
    {
      this->m_loCorner = a_loCorner;
      this->m_hiCorner = a_hiCorner;
    }

    /*!
      @brief Destructor
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline ~BoxSDF() noexcept
    {}

    /*!
      @brief Signed distance function for sphere.
      @param[in] a_point Position.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Real
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

    /*!
      @brief Factory method for building the implicit function on the GPU.
    */
#ifdef EBGEOMETRY_ENABLE_GPU
    EBGEOMETRY_GPU_HOST
    [[nodiscard]] inline void*
    putOnGPU() const noexcept override
    {
      GPUPointer<BoxSDF> box = allocateImplicitFunctionOnDevice<BoxSDF>();

      Vec3* loCorner;
      Vec3* hiCorner;

      cudaMalloc((void**)&loCorner, sizeof(Vec3));
      cudaMalloc((void**)&hiCorner, sizeof(Vec3));

      cudaMemcpy(loCorner, &m_loCorner, sizeof(Vec3), cudaMemcpyHostToDevice);
      cudaMemcpy(hiCorner, &m_hiCorner, sizeof(Vec3), cudaMemcpyHostToDevice);

      createImplicitFunctionOnDevice<<<1, 1>>>(box, loCorner, hiCorner);

      return box;
    }
#endif

  protected:
    /*!
      @brief Low box corner
    */
    Vec3 m_loCorner;

    /*!
      @brief High box corner
    */
    Vec3 m_hiCorner;
  };

} // namespace EBGeometry

#endif
