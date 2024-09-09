/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_GPUTypes.hpp
  @brief  Declaration of various useful implementation of std-like classes for GPUs
  @author Robert Marskar
*/

#ifndef EBGeometry_GPUTypes
#define EBGeometry_GPUTypes

// Our includes
#include "EBGeometry_Types.hpp"

namespace EBGeometry {

  /*!
    @brief Minimum operation of two numbers.
    @param[in] x Number to compare.
    @param[in] y Number to compare.     
  */
  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] inline Real
  min(const Real& x, const Real& y) noexcept
  {
    return x <= y ? x : y;
  }

  /*!
    @brief Maximum operation of two numbres.
    @param[in] x Number to compare.
    @param[in] y Number to compare.     
  */
  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] inline Real
  max(const Real& x, const Real& y) noexcept
  {
    return x >= y ? x : y;
  }

  /*!
    @brief Various useful limits so that we can use numeric_limits<>-like functionality on the GPU.
  */
  namespace Limits {
    /*!
      @brief Maximum representable number.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Real
    max() noexcept
    {
      return 1.E99;
    }

    /*!
      @brief Minium representable number.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Real
    min() noexcept
    {
      return 1.E-99;
    }

    /*!
      @brief Lowest representable number.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Real
    lowest() noexcept
    {
      return -1.E99;
    }
  } // namespace Limits
} // namespace EBGeometry

#endif
