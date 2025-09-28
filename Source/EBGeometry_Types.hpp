// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   EBGeometry_Types.hpp
 * @author Robert Marskar
 * @brief  Various compile-time specifications for EBGeometry
 */

#ifndef EBGEOMETRY_TYPES_HPP
#define EBGEOMETRY_TYPES_HPP

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
} // namespace EBGeometry
#endif

#endif
