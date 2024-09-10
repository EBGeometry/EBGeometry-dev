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

#include "EBGeometry_Types.hpp"
#include "EBGeometry_GPUTypes.hpp"
#include "EBGeometry_Vec.hpp"

namespace EBGeometry {

  /*!
      @brief Encodable SFC concept -- class must have a static function static uint64_t encode(const Index&). This is the main interface for SFCs
    */
  template <typename ImpFunc>
  concept ImplicitFunction = requires(const ImpFunc& func, const Vec3& x) {
    { func.template operator()(x) } -> std::same_as<Real>;
  };
} // namespace EBGeometry

#endif
