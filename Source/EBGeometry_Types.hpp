/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_Types.hpp
  @brief  Compile-time precision specification
  @author Robert Marskar
*/

#ifndef EBGeometry_Types
#define EBGeometry_Types

#include <cfloat>

// Floating-point precision checks
#if !defined(EBGEOMETRY_USE_DOUBLE) && !defined(EBGEOMETRY_USE_FLOAT)
#error "Exactly one of EBGeometry_USE_DOUBLE or EBGeometry_USE_FLOAT must be defined"
#endif

#if defined(EBGEOMETRY_USE_DOUBLE) && defined(EBGEOMETRY_USE_FLOAT)
#error "Only one of EBGEOMETRY_USE_DOUBLE or EBGEOMETRY_USE_FLOAT must be defined"
#endif

#ifdef EBGEOMETRY_USE_DOUBLE
namespace EBGeometry {
  using Real = double;

  constexpr Real MaximumReal = DBL_MAX;
  constexpr Real MinimumReal = DBL_MIN;
  constexpr Real LowestReal  = -DBL_MAX;
} // namespace EBGeometry
#endif

#ifdef EBGEOMETRY_USE_FLOAT
namespace EBGeometry {
  using Real                 = float;
  constexpr Real MaximumReal = FLT_MAX;
  constexpr Real MinimumReal = FLT_MIN;
  constexpr Real LowestReal  = -FLT_MAX;
} // namespace EBGeometry
#endif

#endif
