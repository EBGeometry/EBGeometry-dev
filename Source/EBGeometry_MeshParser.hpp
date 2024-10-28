/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_MeshParser.hpp
  @brief  Utility functions for reading files into EBGeometry data structures
  @author Robert Marskar
*/

#ifndef EBGeometry_MeshParser
#define EBGeometry_MeshParser

// Std includes
#include <vector>

// Our includes
#include "EBGeometry_DCEL_Mesh.hpp"
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_GPUTypes.hpp"
#include "EBGeometry_Macros.hpp"
#include "EBGeometry_Vec.hpp"

namespace EBGeometry {

  /*!
    @brief Namespace for parsing files into various distance functions representations. None of these functions
    are callabled on the device.
  */
  namespace MeshParser {

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
    template <typename Meta = DCEL::DefaultMetaData>
    EBGEOMETRY_GPU_HOST [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE static EBGeometry::DCEL::Mesh<Meta>*
    readIntoDCEL(const std::string a_filename) noexcept;

    /*!
      @brief Get file type
      @param[in] a_filenames 
    */
    EBGEOMETRY_GPU_HOST
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE static MeshParser::FileType
    getFileType(const std::string a_filename) noexcept;

    /*!
      @brief Compress polygon soup. This removes degenerate polygons (e.g., triangles).
      @details This will iterate through 
      @param[in, out] a_vertices Vertices
      @param[in, out] a_polygons Planar polygons.
    */
    EBGEOMETRY_GPU_HOST
    EBGEOMETRY_ALWAYS_INLINE static void
    removeDegeneratePolygonsFromSoup(std::vector<EBGeometry::Vec3>& a_vertices,
                                     std::vector<std::vector<int>>& a_polygons) noexcept;

  } // namespace MeshParser
} // namespace EBGeometry

#include "EBGeometry_MeshParserImplem.hpp"

#endif
