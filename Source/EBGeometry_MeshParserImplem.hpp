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
#include <map>
#include <utility>
#include <fstream>
#include <sstream>
#include <cstring>

// Our includes
#include "EBGeometry_Macros.hpp"
#include "EBGeometry_MeshParser.hpp"
#include "EBGeometry_MeshParser_STL.hpp"

namespace EBGeometry {

  template <typename Meta>
  EBGEOMETRY_ALWAYS_INLINE EBGeometry::DCEL::Mesh<Meta>*
                           MeshParser::readIntoDCEL(const std::string a_filename) noexcept

  {
#warning "MeshParser::readIntoDCEL -- readIntoDCEL not implemented"

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

  EBGEOMETRY_ALWAYS_INLINE bool
  MeshParser::containsDegeneratePolygons(const std::vector<EBGeometry::Vec3>& a_vertices,
                                         const std::vector<std::vector<int>>& a_polygons) noexcept
  {
    EBGEOMETRY_ALWAYS_EXPECT(a_vertices.size() >= 3);
    EBGEOMETRY_ALWAYS_EXPECT(a_polygons.size() >= 1);

    for (const auto& polygon : a_polygons) {
      std::vector<Vec3> polyVertices;

      for (const auto& i : polygon) {
        polyVertices.emplace_back(a_vertices[i]);
      }

      std::sort(polyVertices.begin(), polyVertices.end(), [](const Vec3& a, const Vec3& b) { return a.lessLX(b); });

      for (int i = 1; i < polyVertices.size(); i++) {
        const Vec3& curVertex  = polyVertices[i];
        const Vec3& prevVertex = polyVertices[i - 1];

        if (curVertex == prevVertex) {
          return true;
        }
      }
    }

    return false;
  }

  EBGEOMETRY_ALWAYS_INLINE void
  MeshParser::removeDegenerateVerticesFromSoup(std::vector<EBGeometry::Vec3>& a_vertices,
                                               std::vector<std::vector<int>>& a_polygons) noexcept
  {
    // Polygon soups might contain degenerate vertices. For example, STL files define individual facets rather
    // than links between vertices, and in this case many vertices will be degenerate. Here, we create a map of
    // the original vertices, and delete vertices that are degenerate. The polygons are rearranged so that they
    // reference unique vertices.
    EBGEOMETRY_ALWAYS_EXPECT(a_vertices.size() >= 3);
    EBGEOMETRY_ALWAYS_EXPECT(a_polygons.size() >= 1);

    std::vector<std::pair<Vec3, int>> vertexMap;
    std::map<int, int>                indexMap;

    // Create a map of the original vertices and sort it lexicographically.
    for (int i = 0; i < a_vertices.size(); i++) {
      vertexMap.emplace_back(a_vertices[i], i);
    }

    std::sort(vertexMap.begin(), vertexMap.end(), [](const std::pair<Vec3, int>& A, const std::pair<Vec3, int>& B) {
      const Vec3& a = A.first;
      const Vec3& b = B.first;

      return a.lessLX(b);
    });

    // Compress the vertex vector, and rebuild the index map
    a_vertices.resize(0);

    a_vertices.emplace_back(vertexMap.front().first);
    indexMap.emplace(vertexMap.front().second, 0);

    for (int i = 1; i < vertexMap.size(); i++) {
      const auto& oldIndex = vertexMap[i].second;
      const auto& curVert  = vertexMap[i].first;
      const auto& prevVert = vertexMap[i - 1].first;

      if (curVert != prevVert) {
        a_vertices.emplace_back(curVert);
      }

      indexMap.emplace(oldIndex, a_vertices.size() - 1);
    }

    // Update polygon indexing.
    for (int i = 0; i < a_polygons.size(); i++) {
      std::vector<int>& polygon = a_polygons[i];

      EBGEOMETRY_EXPECT(polygon.size() >= 3);

      for (int v = 0; v < polygon.size(); v++) {
        EBGEOMETRY_EXPECT(polygon[v] >= 0);

        polygon[v] = indexMap.at(polygon[v]);
      }
    }
  }

  template <typename MetaData>
  EBGEOMETRY_ALWAYS_INLINE EBGeometry::DCEL::Mesh<MetaData>
                           MeshParser::turnPolygonSoupIntoDCEL(const std::vector<EBGeometry::Vec3>& a_vertices,
                                      const std::vector<std::vector<int>>& a_polygons) noexcept
  {
#warning "MeshParser::turnPolygonSoupIntoDCEL -- not implemented"
  }
} // namespace EBGeometry

#endif
