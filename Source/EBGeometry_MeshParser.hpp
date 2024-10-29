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
    template <typename MetaData>
    EBGEOMETRY_GPU_HOST
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    static EBGeometry::DCEL::Mesh<MetaData>*
    readIntoDCEL(const std::string a_filename) noexcept;

    /*!
      @brief Get file type
      @param[in] a_filenames 
    */
    EBGEOMETRY_GPU_HOST
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    static MeshParser::FileType
    getFileType(const std::string a_filename) noexcept;

    /*!
      @brief Check if polygons in a polygon soup contain degenerate vertices
      @param[out] a_vertices Vertices
      @param[out] a_polygons Polygons
    */
    EBGEOMETRY_GPU_HOST
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    static bool
    containsDegeneratePolygons(const std::vector<EBGeometry::Vec3>& a_vertices,
                               const std::vector<std::vector<int>>& a_polygons) noexcept;

    /*!
      @brief Compress polygon soup. This removes degenerate polygons (e.g., triangles).
      @details This will iterate through 
      @param[in, out] a_vertices Vertices
      @param[in, out] a_polygons Planar polygons.
    */
    EBGEOMETRY_GPU_HOST
    EBGEOMETRY_ALWAYS_INLINE
    static void
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
    EBGEOMETRY_GPU_HOST
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    static EBGeometry::DCEL::Mesh<MetaData>*
    turnPolygonSoupIntoDCEL(const std::vector<EBGeometry::Vec3>& a_vertices,
                            const std::vector<std::vector<int>>& a_polygons) noexcept;

    /*!
      @brief MeshParser class for reading STL files into DCEL mesh files. 
    */
    class STL
    {
    public:
      /*!
	@brief Read a single STL object from the input file. The file can be binary or ASCII. 
	If the STL file contains multiple solids, this routine returns the first one. 
	@param[in] a_fileName STL file name.
	@return Returns a DCEL mesh
      */
      template <typename MetaData>      
      EBGEOMETRY_GPU_HOST
      [[nodiscard]] EBGEOMETRY_INLINE
      static EBGeometry::DCEL::Mesh<MetaData>*
      readSingle(const std::string a_fileName) noexcept;

      /*!
	@brief Read a single STL object from the input file. The file can be binary or ASCII.
	@param[in] a_fileName STL file name.
	@return Returns DCEL meshes for each object in the input file. The returned strings are identifiers
	for the STL objects.
      */
      template <typename MetaData>
      EBGEOMETRY_GPU_HOST
      [[nodiscard]] EBGEOMETRY_INLINE
      static std::vector<std::pair<EBGeometry::DCEL::Mesh<MetaData>*, std::string>>
      readMulti(const std::string a_fileName) noexcept;

    protected:
      /*!
	@brief Check if the input STL file is an ASCII file or a binary
	@param[in] a_filename File name
	@return Returns FileEncoding::ASCII or FileEncoding::Binary,
      */
      EBGEOMETRY_GPU_HOST
      [[nodiscard]] EBGEOMETRY_INLINE
      static MeshParser::FileEncoding
      getEncoding(const std::string a_filename) noexcept;

      /*!
	@brief ASCII reader STL files, possibly containing multiple objects. Each object becomes a DCEL mesh
	@param[in] a_filename Input filename
      */
      template <typename MetaData>
      EBGEOMETRY_GPU_HOST
      [[nodiscard]] EBGEOMETRY_INLINE
      static std::vector<std::pair<EBGeometry::DCEL::Mesh<MetaData>*, std::string>>
      readASCII(const std::string a_filename) noexcept;

      /*!
	@brief Binary reader for STL files, possibly containing multiple objects. Each object becomes a DCEL mesh
	@param[in] a_filename Input filename
      */
      template <typename MetaData>
      EBGEOMETRY_GPU_HOST
      [[nodiscard]] EBGEOMETRY_INLINE
      static std::vector<std::pair<EBGeometry::DCEL::Mesh<MetaData>*, std::string>>
      readBinary(const std::string a_filename) noexcept;

      /*!
	@brief Read an STL object as a triangle soup into a raw vertices and facets
	@param[out] a_vertices   Vertices
	@param[out] a_facets     STL facets
	@param[out] a_objectName Object name
	@param[out] a_fileContents File contents
	@param[out] a_firstLine  Line number in a_filename containing the 'solid' identifier. 
	@param[out] a_lastLine   Line number in a_filename containing the 'endsolid' identifier. 
      */
      EBGEOMETRY_GPU_HOST
      EBGEOMETRY_INLINE
      static void
      readSTLSoupASCII(std::vector<Vec3>&              a_vertices,
                       std::vector<std::vector<int>>&  a_facets,
                       std::string&                    a_objectName,
                       const std::vector<std::string>& a_fileContents,
                       const int                       a_firstLine,
                       const int                       a_lastLine) noexcept;
    };    

  } // namespace MeshParser
} // namespace EBGeometry

#include "EBGeometry_MeshParserImplem.hpp"

#endif
