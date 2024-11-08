/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file    EBGeometry_ImplicitFunction.hpp
  @brief   Declaration of the implicit function interface
  @author  Robert Marskar
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

  /*!
    @brief Implicit function base class. 
    @details An implicit function must have the following properties:

    1. A member function ::value(const Vec3& x) const which returns the distance (or value function) to an object
  */
  class ImplicitFunction
  {
  public:
    /*!
      @brief Base constructor.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    ImplicitFunction() noexcept
    {}

    /*!
      @brief Destructor
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    virtual ~ImplicitFunction() noexcept
    {}

    /*!
      @brief Value function.
      @details Must return values > 0 if the input point is on the outside of the object and
      values < 0 if the input point is on the inside of the object.
      @param[in] a_point Physical point in space.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] virtual Real
    value(const Vec3& a_point) const noexcept = 0;
  };
#ifdef EBGEOMETRY_ENABLE_GPU

  /*!
    @brief A one-liner for building an implicit function on the device. User must
    supply the constructor arguments
  */
  template <typename ImpFunc, typename... Args>
  EBGEOMETRY_GPU_HOST
  EBGEOMETRY_ALWAYS_INLINE
  void
  buildImplicitFunctionOnDevice(ImpFunc** a_implicitFunction, const Args... args)
  {
    allocateImplicitFunctionOnDevice(a_implicitFunction);

    constructImplicitFunctionOnDevice<ImpFunc><<<1, 1>>>(*a_implicitFunction, args...);
  }
  /*!
    @brief Allocate pointer for implicit function on the GPU
    @param[in, out] implicitFunction Implicit function pointer to be allocated. 
  */
  template <typename ImpFunc>
  EBGEOMETRY_GPU_HOST
  EBGEOMETRY_ALWAYS_INLINE
  void
  allocateImplicitFunctionOnDevice(ImpFunc** a_implicitFunction) noexcept
  {
    cudaMalloc((void**)a_implicitFunction, sizeof(ImpFunc));
  }

  /*!
    @brief Function for building an arbitrary implicit function. Used when constructing
    implicit functions on the GPU. The user inputs the implicit function type (T) and the
    constructor arguments required for constructing the function.
    @param[in] a_implicitFunction Implicit function to be created on device
    @param[in] args Constructor arguments.
  */
  template <typename ImpFunc>
  EBGEOMETRY_GPU_GLOBAL void
  freeImplicitFunctionOnDevice(ImpFunc* a_implicitFunction)
  {
    cudaFree(a_implicitFunction);
  };

  /*!
    @brief Function for constructing an arbitrary implicit function on the device. 
    The user inputs the implicit function type and the constructor arguments required
    for constructing the function. 
    @param[in] a_implicitFunction Implicit function to be created on device
    @param[in] args Constructor arguments.
  */
  template <typename ImpFunc, typename... Args>
  EBGEOMETRY_GPU_GLOBAL void
  constructImplicitFunctionOnDevice(ImpFunc* a_implicitFunction, Args... args)
  {
    new (a_implicitFunction) ImpFunc(args...);
  };
#endif
} // namespace EBGeometry

#endif
