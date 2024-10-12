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

#ifdef EBGEOMETRY_USE_DOUBLE
namespace EBGeometry {
  using Real                 = double;
  constexpr Real MaximumReal = DBL_MAX;
  constexpr Real MinimumReal = DBL_MIN;
  constexpr Real LowestReal  = -DBL_MAX;
  constexpr Real Epsilon     = DBL_EPSILON;
} // namespace EBGeometry
#else
namespace EBGeometry {
  using Real                 = float;
  constexpr Real MaximumReal = FLT_MAX;
  constexpr Real MinimumReal = FLT_MIN;
  constexpr Real LowestReal  = -FLT_MAX;
  constexpr Real Epsilon     = FLT_EPSILON;
} // namespace EBGeometry
#endif

#endif
