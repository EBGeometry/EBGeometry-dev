// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    EBGeometry_ImplicitFunction.hpp
 * @brief   Declaration of the implicit function interface
 * @author  Robert Marskar
 */

#ifndef EBGeometry_ImplicitFunction
#define EBGeometry_ImplicitFunction

// Std includes
#include <type_traits>

// Our includes
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_GPUTypes.hpp"
#include "EBGeometry_Macros.hpp"
#include "EBGeometry_Types.hpp"
#include "EBGeometry_Vec.hpp"

namespace EBGeometry {

  /**
   * @brief Implicit function base class.
   * @details An implicit function must have the following properties:
   *
   * 1. A member function ::value(const Vec3& x) const which returns the distance (or value function) to an object
   */
  class ImplicitFunction
  {
  public:
    /**
     * @brief Default constructor.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    ImplicitFunction() noexcept = default;

    /**
     * @brief Copy constructor.
     * @param[in] a_implicitFunction Other implicit function
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    ImplicitFunction(const ImplicitFunction& a_implicitFunction) noexcept = default;

    /**
     * @brief Move constructor.
     * @param[in] a_implicitFunction Other implicit function
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    ImplicitFunction(ImplicitFunction&& a_implicitFunction) noexcept = default;

    /**
     * @brief Destructor
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    virtual ~ImplicitFunction() noexcept = default;

    /**
     * @brief Copy assignment.
     * @param[in] a_implicitFunction Other implicit function
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    ImplicitFunction&
    operator=(const ImplicitFunction& a_implicitFunction) noexcept = default;

    /**
     * @brief Move assignment.
     * @param[in] a_implicitFunction Other implicit function
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    ImplicitFunction&
    operator=(ImplicitFunction&& a_implicitFunction) noexcept = default;

    /**
     * @brief Value function.
     * @details Must return values > 0 if the input point is on the outside of the object and
     * values < 0 if the input point is on the inside of the object.
     * @param[in] a_point Physical point in space.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] virtual Real
    value(const Vec3& a_point) const noexcept = 0;
  };

  /**
   * @brief Function for constructing an arbitrary implicit function on the device.
   * The user inputs the implicit function type and the constructor arguments required
   * for constructing the function.
   * @note This routine uses placement new to assign the object, so storage for the
   * implicit function MUST be preallocated before calling this routine.
   * @param[in] a_implicitFunction Implicit function to be created on device
   * @param[in] args Constructor arguments.
   */
  template <typename ImpFunc, typename... Args>
  EBGEOMETRY_GPU_GLOBAL void
  constructImplicitFunctionOnDevice(ImpFunc* a_implicitFunction, const Args... args)
  {
    EBGEOMETRY_ALWAYS_EXPECT(a_implicitFunction != nullptr);

    new (a_implicitFunction) ImpFunc(args...);
  };

  /**
   * @brief Factory method for building an implicit function on the host or device
   * @details The user specifies the placement through the MemoryLocation template
   * argument. The user must also supply the constructor arguments.
   *
   * For example:
   *
   * PlaneSDF* plane = createImpFunc<PlaneSDF, MemoryLocation::Unified>(Vec3, Vec3)
   *
   * @param[in] a_args Constructor arguments for implicit function.
   * @returns Returns a pointer to the implicit function.
   */
  template <typename ImpFunc, MemoryLocation MemLoc, typename... Args>
  EBGEOMETRY_GPU_HOST
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  ImpFunc*
  createImpFunc(const Args... a_args) noexcept
  {
    ImpFunc* implicitFunction = nullptr;

    switch (MemLoc) {
    case MemoryLocation::Host: {
      implicitFunction = new ImpFunc(a_args...);

      break;
    }
#ifdef EBGEOMETRY_ENABLE_CUDA
    case MemoryLocation::Global: {
      cudaMalloc((void**)&implicitFunction, sizeof(ImpFunc));

      constructImplicitFunctionOnDevice<ImpFunc><<<1, 1>>>(implicitFunction, a_args...);

      break;
    }
    case MemoryLocation::Unified: {
      cudaMallocManaged((void**)&implicitFunction, sizeof(ImpFunc));

      constructImplicitFunctionOnDevice<ImpFunc><<<1, 1>>>(implicitFunction, a_args...);

      break;
    }
#endif
    }

#warning "EBGeometry_ImplicitFunction.hpp - GPU placement only partially supported." // NOLINT

    return implicitFunction;
  }

  /**
   * @brief Free up an implicit function. This works on both host and device.
   * @param[in] a_implicitFunction Implicit function to be freed.
   */
  template <typename ImpFunc>
  EBGEOMETRY_GPU_HOST
  EBGEOMETRY_ALWAYS_INLINE
  void
  freeImpFunc(ImpFunc*& a_implicitFunction)
  {
    EBGEOMETRY_ALWAYS_EXPECT(a_implicitFunction != nullptr);

    if (GPU::isDevicePointer(a_implicitFunction)) {
#if EBGEOMETRY_ENABLE_CUDA
      cudaFree(a_implicitFunction);
#endif
    }
    else {
      delete a_implicitFunction;
    }
  };

} // namespace EBGeometry

#endif
