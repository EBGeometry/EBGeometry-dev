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
#include <map>
#include <tuple>

// Our includes
#include "EBGeometry_DCEL_Mesh.hpp"
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_GPUTypes.hpp"
#include "EBGeometry_Macros.hpp"
#include "EBGeometry_TriangleMesh.hpp"
#include "EBGeometry_Vec.hpp"

#warning "Implement a function that turns a triangle mesh into a DCEL mesh"

namespace EBGeometry::MeshParser {

  /*!
    @brief Enum for separating ASCII and binary files
  */
  enum class FileEncoding // NOLINT
  {
    ASCII,
    Binary,
    Unknown
  };

  /*!
    @brief Supported supported file types
  */
  enum class FileType // NOLINT
  {
    STL,
    PLY,
    Unsupported
  };

  /*!
    @brief Alias for a polygon soup with attached meta data
    @details The soup consists of a list of vertex coordinates (first entry in pair) and a list of
    faces (second entry in the tuple). Each face contains a list of vertex indices. 
  */
  template <typename MetaData>
  using PolygonSoup = std::tuple<std::vector<Vec3>, std::vector<std::pair<std::vector<int>, MetaData>>, std::string>;

  /*!
    @brief Read file into a polygon soup. 
    @param[in] a_fileName File name. 
  */
  template <typename MetaData>
  EBGEOMETRY_GPU_HOST
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  static PolygonSoup<MetaData>
  readIntoSoup(const std::string& a_fileName) noexcept;

  /*!
    @brief Read a file containing a single watertight object and return it as a DCEL mesh. This version
    supports multiple file formats. 
    @param[in] a_fileName File name
  */
  template <typename MetaData>
  EBGEOMETRY_GPU_HOST
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  static EBGeometry::DCEL::Mesh<MetaData>
  readIntoDCEL(const std::string& a_fileName) noexcept;

  /*!
    @brief Get file type
    @param[in] a_fileNames 
  */
  EBGEOMETRY_GPU_HOST
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  static MeshParser::FileType
  getFileType(const std::string& a_fileName) noexcept;

  /*!
    @brief Check if polygons in a polygon soup contain degenerate polygons (i.e., polygons with degenerate vertices)
    @param[in] a_soup Polygon soup
  */
  template <typename MetaData>
  EBGEOMETRY_GPU_HOST
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  static bool
  containsDegeneratePolygons(const PolygonSoup<MetaData>& a_soup) noexcept;

  /*!
    @brief Check if the vertex list in a polygon soup contains degenerate vertices (i.e., two or more vertices with the shared positions)
    @param[in] a_soup Polygon soup
  */
  template <typename MetaData>
  EBGEOMETRY_GPU_HOST
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  static bool
  containsDegenerateVertices(const PolygonSoup<MetaData>& a_soup) noexcept;

  /*!
    @brief Compress polygon soup. This removes degenerate polygons (e.g., triangles).
    @details This will iterate through 
    @param[in, out] a_vertices Vertices
    @param[in, out] a_polygons Planar polygons.
  */
  template <typename MetaData>
  EBGEOMETRY_GPU_HOST
  EBGEOMETRY_ALWAYS_INLINE
  static void
  removeDegenerateVerticesFromSoup(PolygonSoup<MetaData>& a_soup) noexcept;

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
  static EBGeometry::DCEL::Mesh<MetaData>
  createDCELMeshFromSoup(const PolygonSoup<MetaData>& a_soup) noexcept;

  /*!
    @brief Turn a polygon soup into a triangle mesh.
    @details This is a converter function for turning a DCEL mesh into a triangle mesh. This function will
    take an existing DCEL mesh and check if every polygon in the DCEL is indeed a triangle. If it is, we allocate
    a list of triangles which are returned to the caller.
    @param[in] a_dcelMesh Pre-existing DCEL mesh, out of which we extract triangles.
    @return A list of triangles, with associated normal vectors on their vertex and edge normals. 
  */
  template <typename MetaData>
  EBGEOMETRY_GPU_HOST
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  static std::vector<Triangle<MetaData>>
  createTrianglesFromDCELMesh(const EBGeometry::DCEL::Mesh<MetaData>& a_dcelMesh) noexcept;

  /*!
    @brief MeshParser class for reading STL files into polygon soups. 
  */
  class STL
  {
  public:
    /*!
      @brief Read a file and turn it into a polygon soup. The file can be binary or ASCII. 
      @param[in] a_fileName Input file name. 
      @return Returns a polygon soup. 
    */
    template <typename MetaData>
    EBGEOMETRY_GPU_HOST
    [[nodiscard]] EBGEOMETRY_INLINE
    static std::vector<PolygonSoup<MetaData>>
    readIntoPolygonSoup(const std::string& a_fileName) noexcept;

  protected:
    /*!
      @brief Check if the input file is an ASCII file or a binary file. 
      @param[in] a_fileName File name
      @return Returns FileEncoding::ASCII or FileEncoding::Binary,
    */
    EBGEOMETRY_GPU_HOST
    [[nodiscard]] EBGEOMETRY_INLINE
    static MeshParser::FileEncoding
    getEncoding(const std::string& a_fileName) noexcept;

    /*!
      @brief Reader for ASCII files, possibly containing multiple objects. 
      @param[in] a_fileName Input filename
    */
    template <typename MetaData>
    EBGEOMETRY_GPU_HOST
    [[nodiscard]] EBGEOMETRY_INLINE
    static std::vector<PolygonSoup<MetaData>>
    readASCII(const std::string& a_fileName) noexcept;

    /*!
      @brief Reader for binary files, possibly containing multiple objects. 
      @param[in] a_fileName Input filename
    */
    template <typename MetaData>
    EBGEOMETRY_GPU_HOST
    [[nodiscard]] EBGEOMETRY_INLINE
    static std::vector<PolygonSoup<MetaData>>
    readBinary(const std::string& a_fileName) noexcept;
  };

  /*!
    @brief MeshParser class for reading PLY files into polygon soups. 
  */
  class PLY
  {
  public:
    /*!
      @brief Read a file and turn it into a polygon soup. The file can be binary or ASCII. 
      @param[in] a_fileName Input file name. 
      @return Returns a polygon soup. 
    */
    template <typename MetaData>
    EBGEOMETRY_GPU_HOST
    [[nodiscard]] EBGEOMETRY_INLINE
    static PolygonSoup<MetaData>
    readIntoPolygonSoup(const std::string& a_fileName) noexcept;

  protected:
    /*!
      @brief Check if the input file is an ASCII file or a binary file. 
      @param[in] a_fileName File name
      @return Returns FileEncoding::ASCII or FileEncoding::Binary,
    */
    EBGEOMETRY_GPU_HOST
    [[nodiscard]] EBGEOMETRY_INLINE
    static MeshParser::FileEncoding
    getEncoding(const std::string& a_fileName) noexcept;

    /*!
      @brief Reader for ASCII files, possibly containing multiple objects. 
      @param[in] a_fileName Input filename
    */
    template <typename MetaData>
    EBGEOMETRY_GPU_HOST
    [[nodiscard]] EBGEOMETRY_INLINE
    static PolygonSoup<MetaData>
    readASCII(const std::string& a_fileName) noexcept;

    /*!
      @brief Reader for binary files, possibly containing multiple objects. 
      @param[in] a_fileName Input filename
    */
    template <typename MetaData>
    EBGEOMETRY_GPU_HOST
    [[nodiscard]] EBGEOMETRY_INLINE
    static PolygonSoup<MetaData>
    readBinary(const std::string& a_fileName) noexcept;
  };
} // namespace EBGeometry::MeshParser

#include "EBGeometry_MeshParserImplem.hpp" // NOLINT

#endif
