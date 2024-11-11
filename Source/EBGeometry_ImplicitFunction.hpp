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

  /*!
    @brief One-liner for building an implicit function on the host. User must supply
    the constructor arguments.
    @param[in] a_implicitFunction to be allocated. Must be initialized to nullptr
  */
  template <typename ImpFunc, typename... Args>
  EBGEOMETRY_GPU_HOST
  EBGEOMETRY_ALWAYS_INLINE
  void
  allocateImplicitFunctionOnHost(ImpFunc*& a_implicitFunction, const Args... args) noexcept
  {
    EBGEOMETRY_ALWAYS_EXPECT(a_implicitFunction == nullptr);

    a_implicitFunction = new ImpFunc(args...);
  }

  /*!
    @brief A one-liner for building an implicit function on the device. User must
    supply the constructor arguments
  */
  template <typename ImpFunc, typename... Args>
  EBGEOMETRY_GPU_HOST
  EBGEOMETRY_ALWAYS_INLINE
  void
  allocateImplicitFunctionOnDevice(ImpFunc*& a_implicitFunction, Args... args) noexcept
  {
    EBGEOMETRY_ALWAYS_EXPECT(a_implicitFunction == nullptr);

#ifdef EBGEOMETRY_ENABLE_CUDA
    cudaMalloc((void**)&a_implicitFunction, sizeof(ImpFunc));

    constructImplicitFunctionOnDevice<ImpFunc><<<1, 1>>>(a_implicitFunction, args...);
#else
#error "EBGeometry_ImplicitFunction::buildImplicitFunctionOnDevice - unknown GPU support requested"
    a_implicitFunction = nullptr;
#endif
  }

  /*!
    @brief Function for building an arbitrary implicit function. Used when constructing
    implicit functions on the GPU. The user inputs the implicit function type (T) and the
    constructor arguments required for constructing the function.
    @param[in] a_implicitFunction Implicit function to be created on device
    @param[in] args Constructor arguments.
  */
  template <typename ImpFunc>
  EBGEOMETRY_GPU_HOST
  EBGEOMETRY_ALWAYS_INLINE
  void
  freeImplicitFunctionOnDevice(ImpFunc*& a_implicitFunction)
  {
#ifdef EBGEOMETRY_ENABLE_GPU
    EBGEOMETRY_ALWAYS_EXPECT(a_implicitFunction != nullptr);
#endif

#ifdef EBGEOMETRY_ENABLE_CUDA
    cudaFree(a_implicitFunction);
#elif EBGEOMETRY_ENABLE_HIP
#endif

    a_implicitFunction = nullptr;
  };

  /*!
    @brief Function for building an arbitrary implicit function. Used when constructing
    implicit functions on the GPU. The user inputs the implicit function type (T) and the
    constructor arguments required for constructing the function.
    @param[in] a_implicitFunction Implicit function to be created on device
    @param[in] args Constructor arguments.
  */
  template <typename ImpFunc>
  EBGEOMETRY_GPU_HOST
  EBGEOMETRY_ALWAYS_INLINE
  void
  freeImplicitFunctionOnHost(ImpFunc*& a_implicitFunction)
  {
    EBGEOMETRY_ALWAYS_EXPECT(a_implicitFunction != nullptr);

    delete a_implicitFunction;
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
    EBGEOMETRY_ALWAYS_EXPECT(a_implicitFunction != nullptr);

    new (a_implicitFunction) ImpFunc(args...);
  };
} // namespace EBGeometry

#endif
