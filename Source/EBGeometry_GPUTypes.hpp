// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   EBGeometry_GPUTypes.hpp
 * @brief  Declaration of various useful implementation of std-like classes for GPUs
 * @author Robert Marskar
 */

#ifndef EBGeometry_GPUTypes
#define EBGeometry_GPUTypes

// Std includes
#include <cmath>
#include <type_traits>

// Our includes
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_Macros.hpp"
#include "EBGeometry_Types.hpp"

namespace EBGeometry {

  /**
   * @brief Minimum operation of two numbers.
   * @param[in] x Number to compare.
   * @param[in] y Number to compare.
   */
  EBGEOMETRY_GPU_HOST_DEVICE
  template <typename T>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  constexpr T
  min(const T& x, const T& y) noexcept
  {
    return (x <= y) ? x : y;
  }

  /**
   * @brief Maximum operation of two numbers.
   * @param[in] x Number to compare.
   * @param[in] y Number to compare.
   */
  EBGEOMETRY_GPU_HOST_DEVICE
  template <typename T>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  constexpr T
  max(const T& x, const T& y) noexcept
  {
    return (x >= y) ? x : y;
  }

  /**
   * @brief Sign of number. > 1 if positive, < 1 if negative, and 0 if zero.
   * @param[in] x Input number
   */
  EBGEOMETRY_GPU_HOST_DEVICE
  template <typename T>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  constexpr int
  sgn(const T& x) noexcept
  {
    return (x > T(0)) - (x < T(0));
  }

  /**
   * @brief Sign of number. > 1 if positive, < 1 if negative, and 0 if zero.
   * @param[in] x Input number
   */
  EBGEOMETRY_GPU_HOST_DEVICE
  template <typename T>
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  constexpr T
  abs(const T& x) noexcept
  {
    if constexpr (std::is_floating_point_v<T>) {
      return std::fabs(x);
    }
    else if constexpr (std::is_signed_v<T>) {
      return std::abs(x);
    }
    else {
      return x;
    }
  }

  /**
   * @brief Various useful limits so that we can use numeric_limits<>-like functionality on the GPU.
   */
  namespace Limits {
    /**
     * @brief Maximum representable number.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr Real
    max() noexcept
    {
      return EBGeometry::MaximumReal;
    }

    /**
     * @brief Minimum representable number.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr Real
    min() noexcept
    {
      return EBGeometry::MinimumReal;
    }

    /**
     * @brief Lowest representable number.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr Real
    lowest() noexcept
    {
      return EBGeometry::LowestReal;
    }

    /**
     * @brief Machine precision.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr Real
    eps() noexcept
    {
      return EBGeometry::Epsilon;
    }
  } // namespace Limits
} // namespace EBGeometry

#endif
