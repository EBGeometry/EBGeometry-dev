/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_Concepts.hpp
  @brief  Declaration of various concept interfaces
  @author Robert Marskar
*/

#ifndef EBGeometry_Concepts
#define EBGeometry_Concepts

namespace EBGeometry {
  namespace SFC {
    /*!
    @brief Concept for a space-filling curve. Encodable SFC concept -- class must have a static function static uint64_t encode(const Index&). This is the main interface for SFCs
  */
    template <typename S>
    concept Encodable = requires(const Index& point, const SFC::Code code) {
      { S::encode(point) } -> std::same_as<SFC::Code>;
      { S::decode(code) } -> std::same_as<Index>;
    };
  }

#endif
