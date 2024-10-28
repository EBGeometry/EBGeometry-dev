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
      @brief Read a file containing a single watertight object and return it as a DCEL mesh. This version
      supports multiple file formats. 
      @param[in] a_filename File name
    */
    template <typename MetaData = DCEL::DefaultMetaData>
    EBGEOMETRY_GPU_HOST [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE static EBGeometry::DCEL::Mesh<MetaData>*
    readIntoDCEL(const std::string a_filename) noexcept;

    /*!
      @brief Get file type
      @param[in] a_filenames 
    */
    EBGEOMETRY_GPU_HOST
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE static MeshParser::FileType
    getFileType(const std::string a_filename) noexcept;

    /*!
      @brief Check if polygons in a polygon soup contain degenerate vertices
      @param[out] a_vertices Vertices
      @param[out] a_polygons Polygons
    */
    EBGEOMETRY_GPU_HOST
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE static bool
    containsDegeneratePolygons(const std::vector<EBGeometry::Vec3>& a_vertices,
                               const std::vector<std::vector<int>>& a_polygons) noexcept;

    /*!
      @brief Compress polygon soup. This removes degenerate polygons (e.g., triangles).
      @details This will iterate through 
      @param[in, out] a_vertices Vertices
      @param[in, out] a_polygons Planar polygons.
    */
    EBGEOMETRY_GPU_HOST
    EBGEOMETRY_ALWAYS_INLINE static void
    removeDegenerateVerticesFromSoup(std::vector<EBGeometry::Vec3>& a_vertices,
                                     std::vector<std::vector<int>>& a_polygons) noexcept;

    /*!
      @brief Turn raw vertices into a DCEL mesh.
      @details The input vector of vertices contains the coordinates of each vertex. The polygon list
      contains the list of polygons (outer vector), where each entry contains a list of vertices (inner vector)
      that describe which vertices make up the polygon. 
      @param[in]  a_vertices Vertex list.
      @param[in]  a_polygons POlygon list. 
      @return Returns an allocated DCEL mesh. It is up to the user to properly free memory from this mesh when it
      is no longer required. 
    */
    template <typename MetaData>
    EBGEOMETRY_GPU_HOST [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE static EBGeometry::DCEL::Mesh<MetaData>
    turnPolygonSoupIntoDCEL(const std::vector<EBGeometry::Vec3>& a_vertices,
                            const std::vector<std::vector<int>>& a_polygons) noexcept;

  } // namespace MeshParser
} // namespace EBGeometry

#include "EBGeometry_MeshParserImplem.hpp"

#endif
