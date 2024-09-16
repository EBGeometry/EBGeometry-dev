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
#include <concepts>

// Our includes
#include "EBGeometry_Types.hpp"
#include "EBGeometry_GPUTypes.hpp"
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

    /*!
      @brief Factory method for building the implicit function on the GPU. Returns a pointer to the
      implicit function on the GPU. 
      @details This function must create another implicit function on the GPU -- it is
      exceptionally important that implementations do this since vtables are not copyable from host to device.
    */
#ifdef EBGEOMETRY_ENABLE_GPU
    EBGEOMETRY_GPU_HOST
    [[nodiscard]] virtual void*
    putOnGPU() const noexcept = 0;
#endif
  };

  /*!
    @brief Parent class for building implicit functions
  */
  template <typename T>
  class ImplicitFunctionFactory
  {
  public:
    /*!
      @brief Build implicit function on host
    */
    EBGEOMETRY_GPU_HOST
    virtual T*
    buildOnHost() const noexcept = 0;

    /*!
      @brief Build implicit function on the device
    */
    EBGEOMETRY_GPU_HOST
    virtual GPUPointer<T>
    buildOnDevice() const noexcept = 0;
  };
} // namespace EBGeometry

#endif
