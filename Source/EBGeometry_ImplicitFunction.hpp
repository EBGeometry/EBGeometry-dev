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
    @brief Factory method for building an implicit function on the host or device
    @details The user specifies the placement through the MemoryLocation template
    argument. The user must also supply the constructor arguments.

    For example:

    PlaneSDF* plane = createImpFunc<PlaneSDF, MemoryLocation::Unified>(Vec3, Vec3)
    
    @param[in] a_implicitFunction to be allocated. Must be initialized to nullptr
    @returns Returns a pointer to the implicit function.
  */
  template <typename ImpFunc, MemoryLocation MemLoc, typename... Args>
  EBGEOMETRY_GPU_HOST
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  ImpFunc*
  createImpFunc(const Args... args) noexcept
  {
    ImpFunc* implicitFunction = nullptr;

    switch (MemLoc) {
    case MemoryLocation::Host: {
      implicitFunction = new ImpFunc(args...);

      break;
    }
#ifdef EBGEOMETRY_ENABLE_CUDA
    case MemoryLocation::Global: {
      cudaMalloc((void**)&implicitFunction, sizeof(ImpFunc));

      constructImplicitFunctionOnDevice<ImpFunc><<<1, 1>>>(implicitFunction, args...);

      break;
    }
    case MemoryLocation::Unified: {
      cudaMallocManaged((void**)&implicitFunction, sizeof(ImpFunc));

      constructImplicitFunctionOnDevice<ImpFunc><<<1, 1>>>(implicitFunction, args...);

      break;
    }
#endif
    }

#warning "EBGeometry_ImplicitFunction.hpp - GPU placement is only partially supported"

    return implicitFunction;
  }

  /*!
    @brief Free up an implicit function. This works on both host and device.
    @param[in] a_implicitFunction Implicit function to be freed.
  */
  template <typename ImpFunc>
  EBGEOMETRY_GPU_HOST
  EBGEOMETRY_ALWAYS_INLINE
  void
  freeImpFunc(ImpFunc*& a_implicitFunction)
  {
    EBGEOMETRY_ALWAYS_EXPECT(a_implicitFunction != nullptr);

#ifdef EBGEOMETRY_ENABLE_CUDA
    cudaPointerAttributes attr;

    cudaPointerGetAttributes(&attr, a_implicitFunction);

    if ((attr.type == cudaMemoryTypeHost) || (attr.type == cudaMemoryTypeUnregistered)) {
      delete a_implicitFunction;
    }
    else if ((attr.type == cudaMemoryTypeDevice) || (attr.type == cudaMemoryTypeManaged)) {
      cudaFree(a_implicitFunction);
    }
#else
    delete a_implicitFunction;
#endif
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
  constructImplicitFunctionOnDevice(ImpFunc* a_implicitFunction, const Args... args)
  {
    EBGEOMETRY_ALWAYS_EXPECT(a_implicitFunction != nullptr);

    new (a_implicitFunction) ImpFunc(args...);
  };
} // namespace EBGeometry

#endif
