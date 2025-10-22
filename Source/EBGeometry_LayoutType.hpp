// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   EBGeometry_LayoutType.hpp
 * @author Robert Marskar
 * @brief  Definition of layout tags/enums used to select data layout policies.
 */

#ifndef EBGeometry_LayoutType
#define EBGeometry_LayoutType

namespace EBGeometry {

  /**
   * @enum LayoutType
   * @brief Enumerates memory layout policies.
   * @details
   * - `AoS`: Array-of-Structs layout, e.g., contiguous array of Triangle objects.
   * - `SoA`: Structure-of-Arrays layout, e.g., separate contiguous arrays for each field.
   */
  enum class LayoutType
  {
    AoS, ///< Array-of-Structs layout.
    SoA  ///< Structure-of-Arrays layout.
  };

} // namespace EBGeometry

#endif // EBGeometry_LayoutType
