// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   EBGeometry_Types.hpp
 * @author Robert Marskar
 * @brief  Compile-time precision specification for EBGeometry.
 *
 * This header defines the floating-point type (`Real`) and associated
 * compile-time constants used within EBGeometry. The precision (single
 * or double) is determined at compile time via the `EBGEOMETRY_USE_DOUBLE`
 * preprocessor flag.
 *
 * - If `EBGEOMETRY_USE_DOUBLE` is defined, `Real` is `double`.
 * - Otherwise, `Real` is `float`.
 *
 * The file also provides maximum, minimum, lowest, and epsilon constants
 * for the selected type.
 *
 * @author Robert Marskar
 */

#ifndef EBGeometry_Types
#define EBGeometry_Types

#include <cfloat>

/**
 * @namespace EBGeometry
 * @brief Namespace containing types and constants for EBGeometry.
 */
#ifdef EBGEOMETRY_USE_DOUBLE
namespace EBGeometry {
  /**
   * @typedef Real
   * @brief Floating-point type used throughout EBGeometry.
   *
   * Defined as `double` when `EBGEOMETRY_USE_DOUBLE` is enabled.
   */
  using Real = double;

  /**
   * @brief Maximum representable value of `Real`.
   */
  constexpr Real MaximumReal = DBL_MAX;

  /**
   * @brief Minimum positive normalized value of `Real`.
   */
  constexpr Real MinimumReal = DBL_MIN;

  /**
   * @brief Lowest finite value of `Real` (most negative).
   */
  constexpr Real LowestReal = -DBL_MAX;

  /**
   * @brief Difference between 1.0 and the next representable value of `Real`.
   */
  constexpr Real Epsilon = DBL_EPSILON;
} // namespace EBGeometry
#else
namespace EBGeometry {
  /**
   * @typedef Real
   * @brief Floating-point type used throughout EBGeometry.
   *
   * Defined as `float` when `EBGEOMETRY_USE_DOUBLE` is not enabled.
   */
  using Real = float;

  /**
   * @brief Maximum representable value of `Real`.
   */
  constexpr Real MaximumReal = FLT_MAX;

  /**
   * @brief Minimum positive normalized value of `Real`.
   */
  constexpr Real MinimumReal = FLT_MIN;

  /**
   * @brief Lowest finite value of `Real` (most negative).
   */
  constexpr Real LowestReal = -FLT_MAX;

  /**
   * @brief Difference between 1.0 and the next representable value of `Real`.
   */
  constexpr Real Epsilon = FLT_EPSILON;
} // namespace EBGeometry
#endif

#endif
