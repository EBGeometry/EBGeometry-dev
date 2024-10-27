/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_MeshParserImplem.hpp
  @brief  Implementation of EBGeometry_MeshParser.hpp
  @author Robert Marskar
*/

#ifndef EBGeometry_MeshParserImplem
#define EBGeometry_MeshParserImplem

// Our includes
#include "EBGeometry_MeshParser.hpp"

namespace EBGeometry {

  template <typename Meta>
  EBGEOMETRY_ALWAYS_INLINE EBGeometry::DCEL::Mesh<Meta>*
                           MeshParser::readIntoDCEL(const std::string a_filename) noexcept
  {
#warning "MeshParser::readIntoDCEL -- not implemented"

    return nullptr;
  }

  EBGEOMETRY_ALWAYS_INLINE MeshParser::FileType
                           MeshParser::getFileType(const std::string a_filename) noexcept
  {
    const std::string ext = a_filename.substr(a_filename.find_last_of(".") + 1);

    auto fileType = MeshParser::FileType::Unsupported;

    if (ext == "stl" || ext == "STL") {
      fileType = MeshParser::FileType::STL;
    }
    else if (ext == "ply" || ext == "PLY") {
      fileType = MeshParser::FileType::PLY;
    }

    return fileType;
  }

  EBGEOMETRY_ALWAYS_INLINE void
  MeshParser::removeDegeneratePolygonsFromSoup(std::vector<EBGeometry::Vec3>& a_vertices,
                                               std::vector<std::vector<int>>& a_polygons) noexcept
  {
#warning "MeshParser::removeDegeneratePolygonsFromSoup"
  }

} // namespace EBGeometry

#endif
