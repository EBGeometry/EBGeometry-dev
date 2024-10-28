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
      @brief MeshParser class for reading STL files into DCEL mesh files. 
    */
    class STL
    {
    public:
      /*!
	@brief Read a single STL object from the input file. The file can be binary or ASCII. 
	If the STL file contains multiple solids, this routine returns the first one. 
	@param[in] a_filename STL file name.
	@return Returns a DCEL mesh 
      */
      template <typename MetaData = DCEL::DefaultMetaData>
      EBGEOMETRY_GPU_HOST EBGEOMETRY_INLINE static EBGeometry::DCEL::Mesh<MetaData>*
      read(const std::string a_filename) noexcept;

    protected:
      /*!
	@brief Check if the input STL file is an ASCII file or a binary
	@param[in] a_filename File name
	@return Returns FileEncoding::ASCII or FileEncoding::Binary,
      */
      EBGEOMETRY_GPU_HOST
      EBGEOMETRY_INLINE static FileEncoding
      getEncoding(const std::string a_filename) noexcept;
#if 0
      /*!
	@brief ASCII reader STL files, possibly containing multiple objects. Each object becomes a DCEL mesh
	@param[in] a_filename Input filename
      */
      inline static std::vector<std::pair<std::shared_ptr<Mesh>, std::string>>
      readASCII(const std::string a_filename) noexcept;

      /*!
	@brief Binary reader for STL files, possibly containing multiple objects. Each object becomes a DCEL mesh
	@param[in] a_filename Input filename
      */
      inline static std::vector<std::pair<std::shared_ptr<Mesh>, std::string>>
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
      inline static void
      readSTLSoupASCII(std::vector<Vec3>&                a_vertices,
                       std::vector<std::vector<size_t>>& a_facets,
                       std::string&                      a_objectName,
                       const std::vector<std::string>&   a_fileContents,
                       const size_t                      a_firstLine,
                       const size_t                      a_lastLine) noexcept;
#endif
    };
  } // namespace MeshParser
} // namespace EBGeometry

#include "EBGeometry_MeshParserImplem.hpp"

#endif
