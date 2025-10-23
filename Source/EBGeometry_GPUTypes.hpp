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
  template <typename T>
  EBGEOMETRY_GPU_HOST_DEVICE
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
  template <typename T>
  EBGEOMETRY_GPU_HOST_DEVICE
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
  template <typename T>
  EBGEOMETRY_GPU_HOST_DEVICE
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
  template <typename T>
  EBGEOMETRY_GPU_HOST_DEVICE
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
   * @brief Check that input number is close to one
   * @param[in] x Input number
   */
  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  constexpr bool
  nearOne(const Real& x) noexcept
  {
    return (EBGeometry::abs(x) - Real(1)) <= Real(1E-6);
  }

  /**
   * @brief Begin making enclosed functions available to OpenMP target devices.
   *
   * @note Compilers that support OpenMP offloading will see the enclosed declarations
   *       as device-callable. On compilers without OpenMP, this pragma is ignored.
   */
#if defined(_OPENMP)
#pragma omp declare target
#endif

  /**
   * @brief Mark the next routine as device-callable for OpenACC.
   *
   * @note `seq` is appropriate for scalar math wrappers; vector forms can be
   *       added separately if needed. On compilers without OpenACC, this pragma is ignored.
   */
#if defined(_OPENACC)
#pragma acc routine seq
#endif
  /**
   * @brief Cross-platform square root that works on host and device (CUDA, HIP, SYCL, OpenMP, OpenACC).
   *
   * @tparam T A floating-point type: typically `float`, `double`, or `long double`
   *           (subject to backend support for `long double`).
   * @param x Input value.
   * @return The principal square root of `x`.
   *
   * @details
   * - **SYCL**: Calls `sycl::sqrt` so it can be used inside SYCL kernels.
   * - **CUDA/HIP**: Uses unqualified `::sqrt` to pick up device overloads from libdevice/libhip
   *   (and host overloads in host compilation).
   * - **Others / Host**: Falls back to `std::sqrt`.
   *
   * @note On GPUs, `long double` may have limited precision or be unsupported; prefer
   *       `double` where portable full precision is required.
   * @note This function is `constexpr` where the backend permits evaluation at compile time.
   */
  template <class T>
  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  constexpr T
  sqrt(T x) noexcept
  {
#if defined(SYCL_LANGUAGE_VERSION)
    return sycl::sqrt(x);
#elif defined(__CUDACC__) || defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC__)
    using ::sqrt;
    return sqrt(x);
#else
    return std::sqrt(x);
#endif
  }

  /**
   * @brief End of the OpenMP device exposure block.
   * @see sqrt
   */
#if defined(_OPENMP)
#pragma omp end declare target
#endif

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
      return std::numeric_limits<Real>::max();
    }

    /**
     * @brief Minimum representable number.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr Real
    min() noexcept
    {
      return std::numeric_limits<Real>::min();
    }

    /**
     * @brief Lowest representable number.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr Real
    lowest() noexcept
    {
      return std::numeric_limits<Real>::lowest();
    }

    /**
     * @brief Machine precision.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr Real
    eps() noexcept
    {
      return std::numeric_limits<Real>::epsilon();
    }
  } // namespace Limits
} // namespace EBGeometry

#endif
