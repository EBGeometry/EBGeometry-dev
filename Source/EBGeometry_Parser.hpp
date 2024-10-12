/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_Parser.hpp
  @brief  Utility functions for reading files into EBGeometry data structures
  @author Robert Marskar
*/

#ifndef EBGeometry_Parser
#define EBGeometry_Parser

// Our includes
#include "EBGeometry_DCEL_Mesh.hpp"
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_GPUTypes.hpp"
#include "EBGeometry_Macros.hpp"

namespace EBGeometry {
  namespace Parser {

    /*!
      @brief Enum for separating ASCII and binary files
    */
    enum class FileEncoding
    {
      ASCII,
      Binary,
      Unknown
    };

    /*!
      @brief Supported supported file types
    */
    enum class FileType
    {
      STL,
      PLY,
      Unsupported
    };

    /*!
      @brief Read a file containing a single watertight object and return it as a DCEL mesh
      @param[in] a_filename File name
    */
    EBGEOMETRY_GPU_HOST
    template <typename Meta = DCEL::DefaultMetaData>
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE static EBGeometry::DCEL::Mesh<Meta>*
    readIntoDCEL(const std::string a_filename) noexcept;

  } // namespace Parser
} // namespace EBGeometry

#endif
