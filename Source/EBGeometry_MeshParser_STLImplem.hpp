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

// Std includes
#include <utility>
#include <fstream>
#include <sstream>
#include <cstring>

// Our includes
#include "EBGeometry_Macros.hpp"
#include "EBGeometry_MeshInspector.hpp"
#include "EBGeometry_MeshParser.hpp"
#include "EBGeometry_MeshParser_STL.hpp"

namespace EBGeometry {
  namespace MeshParser {

    template <typename MetaData>
    EBGEOMETRY_INLINE EBGeometry::DCEL::Mesh<MetaData>*
                      STL::readSingle(const std::string a_fileName) noexcept
    {
      return ((STL::readMulti(a_fileName)).front()).first;
    }

    template <typename MetaData>
    EBGEOMETRY_INLINE std::vector<std::pair<EBGeometry::DCEL::Mesh<MetaData>*, std::string>>
                      STL::readMulti(const std::string a_fileName) noexcept
    {
      const auto fileEncoding = MeshParser::STL::getEncoding(a_filename);

      EBGEOMETRY_ALWAYS_EXPECT(MeshParser::getFileType(a_fileName) == MeshParser::FileType::STL);
      EBGEOMETRY_ALWAYS_EXPECT(fileEncoding == MeshParser::FileEncoding::ASCII ||
                               fileEncoding == MeshParser::fileEncoding::Binary);

      std::vector<std::pair<EBGeometry::DCEL::Mesh<MetaData>*, std::string>> meshes;

      switch (fileEncoding) {
      case MeshParser::FileEncoding::ASCII: {
        //        meshes = EBGeometry::MeshParser::STL::readASCII(fileName);

        break;
      }
      case MeshParser::FileEncoding::Binary: {
        //        meshes = EBGeometry::MeshParser::STL::readBinary(fileName);

        break;
      }
      }

      return meshes;
    }

    EBGEOMETRY_INLINE FileEncoding
    STL::getEncoding(const std::string a_filename) noexcept
    {
      MeshParser::FileEncoding fileEncoding = MeshParser::FileEncoding::Unknown;

      std::ifstream is(a_filename, std::istringstream::in | std::ios::binary);

      if (is.good()) {
        char chars[256];
        is.read(chars, 256);

        std::string buffer(chars, static_cast<size_t>(is.gcount()));
        std::transform(buffer.begin(), buffer.end(), buffer.begin(), ::tolower);

        // clang-format off
	if(buffer.find("solid") != std::string::npos && buffer.find("\n")    != std::string::npos &&
	   buffer.find("facet") != std::string::npos && buffer.find("normal")!= std::string::npos) {
	
	  fileEncoding = MeshParser::FileEncoding::ASCII;
	}
	else {
	  fileEncoding = MeshParser::FileEncoding::Binary;
	}
        // clang-format on
      }
      else {
        std::cerr << "MeshParser::STL::getEncoding -- could not open file '" + a_filename + "'\n";
      }

      return fileEncoding;
    }
  } // namespace MeshParser
} // namespace EBGeometry

#endif
