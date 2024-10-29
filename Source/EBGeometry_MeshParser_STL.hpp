/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_MeshParser_STL.hpp
  @brief  Declaration of a static utility class for reading STL files.
  @author Robert Marskar
*/

#ifndef EBGeometry_MeshParser_STL
#define EBGeometry_MeshParser_STL

// Std includes
#include <vector>

// Our includes
#include "EBGeometry_DCEL_Mesh.hpp"
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_GPUTypes.hpp"
#include "EBGeometry_Macros.hpp"
#include "EBGeometry_MeshParser.hpp"
#include "EBGeometry_Vec.hpp"

namespace EBGeometry {
  namespace MeshParser {
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

#include "EBGeometry_MeshParser_STLImplem.hpp"

#endif
