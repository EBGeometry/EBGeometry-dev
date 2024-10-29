/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_MeshParserImplem.hpp
  @brief  Implementation of EBGeometry_MeshParser.hpp
  @author Robert Marskar
*/

#ifndef EBGeometry_MeshParser_STLImplem
#define EBGeometry_MeshParser_STLImplem

// Std includes
#include <utility>
#include <fstream>
#include <sstream>
#include <cstring>

// Our includes
#include "EBGeometry_Macros.hpp"
#include "EBGeometry_MeshParser.hpp"
#include "EBGeometry_MeshParser_STL.hpp"

namespace EBGeometry {
  namespace MeshParser {

    template <typename MetaData>
    EBGEOMETRY_INLINE
    EBGeometry::DCEL::Mesh<MetaData>*
    STL::readSingle(const std::string a_fileName) noexcept
    {
      return ((STL::readMulti<MetaData>(a_fileName)).front()).first;
    }

    template <typename MetaData>
    EBGEOMETRY_INLINE
    std::vector<std::pair<EBGeometry::DCEL::Mesh<MetaData>*, std::string>>
    STL::readMulti(const std::string a_fileName) noexcept
    {
      const auto fileEncoding = MeshParser::STL::getEncoding(a_fileName);

      EBGEOMETRY_ALWAYS_EXPECT(MeshParser::getFileType(a_fileName) == MeshParser::FileType::STL);
      EBGEOMETRY_ALWAYS_EXPECT(fileEncoding == MeshParser::FileEncoding::ASCII ||
                               fileEncoding == MeshParser::FileEncoding::Binary);

      std::vector<std::pair<EBGeometry::DCEL::Mesh<MetaData>*, std::string>> meshes;

      switch (fileEncoding) {
      case MeshParser::FileEncoding::ASCII: {
        meshes = EBGeometry::MeshParser::STL::readASCII<MetaData>(a_fileName);

        break;
      }
      case MeshParser::FileEncoding::Binary: {
        meshes = EBGeometry::MeshParser::STL::readBinary<MetaData>(a_fileName);

        break;
      }
      }

      return meshes;
    }

    EBGEOMETRY_INLINE
    MeshParser::FileEncoding
    STL::getEncoding(const std::string a_fileName) noexcept
    {
      MeshParser::FileEncoding fileEncoding = MeshParser::FileEncoding::Unknown;

      std::ifstream is(a_fileName, std::istringstream::in | std::ios::binary);

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
        std::cerr << "MeshParser::STL::getEncoding -- could not open file '" + a_fileName + "'\n";
      }

      return fileEncoding;
    }

    template <typename MetaData>
    EBGEOMETRY_INLINE
    std::vector<std::pair<EBGeometry::DCEL::Mesh<MetaData>*, std::string>>
    STL::readASCII(const std::string a_fileName) noexcept
    {
      std::vector<std::pair<EBGeometry::DCEL::Mesh<MetaData>*, std::string>> meshes;

#warning "EBGeometry_MeshParser_STLImplem.hpp - working on readASCII function"

      return meshes;
    }

    template <typename MetaData>
    EBGEOMETRY_INLINE
    std::vector<std::pair<EBGeometry::DCEL::Mesh<MetaData>*, std::string>>
    STL::readBinary(const std::string a_fileName) noexcept
    {
      EBGEOMETRY_ALWAYS_EXPECT(MeshParser::getFileType(a_fileName) == MeshParser::FileType::STL);
      EBGEOMETRY_ALWAYS_EXPECT(MeshParser::STL::getEncoding(a_fileName) == MeshParser::FileEncoding::Binary);

      std::vector<std::pair<EBGeometry::DCEL::Mesh<MetaData>*, std::string>> meshes;

      using VertexList   = std::vector<Vec3>;
      using TriangleList = std::vector<std::vector<int>>;

      // Read the file.
      std::ifstream is(a_fileName);
      if (is.is_open()) {

        // Read the header and discard that info.
        char header[80];
        is.read(header, 80);

        // Read number of triangles
        char tmp[4];
        is.read(tmp, 4);
        uint32_t numTriangles;
        memcpy(&numTriangles, &tmp, 4);

        std::map<uint16_t, std::pair<VertexList, TriangleList>> verticesAndTriangles;

        // Read triangles into raw vertices and triangles
        char normal[12];
        char v1[12];
        char v2[12];
        char v3[12];
        char att[2];

        float x;
        float y;
        float z;

        uint16_t id;

        for (int i = 0; i < numTriangles; i++) {
          is.read(normal, 12);
          is.read(v1, 12);
          is.read(v2, 12);
          is.read(v3, 12);
          is.read(att, 2);

          char* ptr = nullptr;

          Vec3 v[3];

          ptr = v1;
          memcpy(&x, ptr, 4);
          ptr += 4;
          memcpy(&y, ptr, 4);
          ptr += 4;
          memcpy(&z, ptr, 4);
          v[0] = Vec3(x, y, z);

          ptr = v2;
          memcpy(&x, ptr, 4);
          ptr += 4;
          memcpy(&y, ptr, 4);
          ptr += 4;
          memcpy(&z, ptr, 4);
          v[1] = Vec3(x, y, z);

          ptr = v3;
          memcpy(&x, ptr, 4);
          ptr += 4;
          memcpy(&y, ptr, 4);
          ptr += 4;
          memcpy(&z, ptr, 4);
          v[2] = Vec3(x, y, z);

          memcpy(&id, &att, 2);

          if (verticesAndTriangles.find(id) == verticesAndTriangles.end()) {
            verticesAndTriangles.emplace(id, std::make_pair(VertexList(), TriangleList()));
          }

          auto& objectVertices = verticesAndTriangles.at(id).first;
          auto& objectFacets   = verticesAndTriangles.at(id).second;

          // Insert a new triangle.
          std::vector<int> curFacet;
          for (int j = 0; j < 3; j++) {
            objectVertices.emplace_back(v[j]);
            curFacet.emplace_back(objectVertices.size() - 1);
          }

          objectFacets.emplace_back(curFacet);
        }

        // Turn the triangle soup into a mesh.
        int curID = 0;
        for (auto& soup : verticesAndTriangles) {
          auto& vertices  = soup.second.first;
          auto& triangles = soup.second.second;

          // Remove degenerate vertices and then make the triangle soup into a DCEL mesh.
          EBGEOMETRY_ALWAYS_EXPECT(!(MeshParser::containsDegeneratePolygons(vertices, triangles)));

          MeshParser::removeDegenerateVerticesFromSoup(vertices, triangles);

          const auto mesh  = MeshParser::turnPolygonSoupIntoDCEL<MetaData>(vertices, triangles);
          const auto strID = std::to_string(curID);

          meshes.emplace_back(mesh, strID);

          curID++;
        }
      }
      else {
        std::cerr << "EBGeometry::MeshParser::STL::readBinary -- could not open file '" + a_fileName + "'\n";
      }

      return meshes;
    }

    EBGEOMETRY_INLINE
    void
    STL::readSTLSoupASCII(std::vector<Vec3>&              a_vertices,
                          std::vector<std::vector<int>>&  a_facets,
                          std::string&                    a_objectName,
                          const std::vector<std::string>& a_fileContents,
                          const int                       a_firstLine,
                          const int                       a_lastLine) noexcept
    {
#warning "EBGeometry_MeshParser_STLImplem.hpp - working on readSTLSoupASCII function"
    }
  } // namespace MeshParser
} // namespace EBGeometry

#endif
