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
    @brief Minimum operation that can be used on GPUs
    @param[in] x Number to compare.
    @param[in] y Number to compare.     
  */
  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] inline Real min(const Real& x, const Real& y) noexcept {
    return x <= y ? x : y;
  }

  /*!
    @brief Maximum operation that can be used on GPUs
    @param[in] x Number to compare.
    @param[in] y Number to compare.     
  */
  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] inline Real max(const Real& x, const Real& y) noexcept {
    return x >= y ? x : y;
  }    
}

#endif
