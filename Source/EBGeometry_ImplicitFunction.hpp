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
#include <memory>

// Our includes
#include "EBGeometry_Types.hpp"
#include "EBGeometry_GPUTypes.hpp"
#include "EBGeometry_Vec.hpp"

namespace EBGeometry {

#ifdef EBGEOMETRY_ENABLE_GPU  
  /*!
    @brief Shortcut for GPU pointer. Note that this is mostly used so that users
    know from the typename which pointers are intended to be pointers to data allocated
    on the device.
  */
  template <typename T>
  using GPUPointer = T**;
#endif


#ifdef EBGEOMETRY_ENABLE_GPU
  /*!
    @brief Allocate pointer for implicit function on the GPU
    @param[inout] implicitFunction Implicit function pointer to be allocated. 
  */
  template <typename T>
  EBGEOMETRY_GPU_HOST inline GPUPointer<T>
  allocateImplicitFunctionOnDevice() noexcept
  {
    GPUPointer<T> func;

    cudaMalloc((void**)&func, sizeof(GPUPointer<T>));

    return func;
  }
#endif

#ifdef EBGEOMETRY_ENABLE_GPU
  /*!
    @brief Function for building an arbitrary implicit function. Used when constructing
    implicit functions on the GPU. The user inputs the implicit function type (T) and the
    constructor arguments required for constructing the function.
    @param[in] a_implicitFunction Implicit function to be created on device
    @param[in] args Constructor arguments.
  */
  template <typename T, typename... Args>
  EBGEOMETRY_GPU_GLOBAL void
  createImplicitFunctionOnDevice(GPUPointer<T> a_implicitFunction, Args... args)
  {
    (*a_implicitFunction) = new T(*args...);
  };
#endif    

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
    inline ImplicitFunction() noexcept
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
    @brief Parent class for building implicit functions
  */
  template <typename T>
  class ImplicitFunctionFactory
  {
  public:

    static_assert(std::is_base_of<ImplicitFunction, T>::value, "T is not a base of ImplicitFunction");
    
    /*!
      @brief Build implicit function on host
    */
    EBGEOMETRY_GPU_HOST
    virtual std::shared_ptr<T>
    buildOnHost() const noexcept = 0;

#ifdef EBGEOMETRY_ENABLE_GPU
    /*!
      @brief Build implicit function on the device
    */
    EBGEOMETRY_GPU_HOST
    virtual GPUPointer<T>
    buildOnDevice() const noexcept = 0;
#endif
  };
} // namespace EBGeometry

#endif
