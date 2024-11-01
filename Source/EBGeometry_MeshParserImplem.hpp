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
#include <cstdint>

// Our includes
#include "EBGeometry_Macros.hpp"
#include "EBGeometry_MeshParser.hpp"

namespace EBGeometry {
  namespace MeshParser {

    EBGEOMETRY_ALWAYS_INLINE
    FileType
    getFileType(const std::string a_fileName) noexcept
    {
      const std::string ext = a_fileName.substr(a_fileName.find_last_of(".") + 1);

      auto fileType = MeshParser::FileType::Unsupported;

      if (ext == "stl" || ext == "STL") {
        fileType = MeshParser::FileType::STL;
      }
      else if (ext == "ply" || ext == "PLY") {
        fileType = MeshParser::FileType::PLY;
      }

      return fileType;
    }

    EBGEOMETRY_ALWAYS_INLINE
    bool
    containsDegeneratePolygons(const std::vector<EBGeometry::Vec3>& a_vertices,
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

    EBGEOMETRY_ALWAYS_INLINE
    void
    removeDegenerateVerticesFromSoup(std::vector<EBGeometry::Vec3>& a_vertices,
                                     std::vector<std::vector<int>>& a_polygons) noexcept
    {
      // Polygon soups might contain degenerate vertices. For example, STL files define individual facets rather
      // than links between vertices, and in this case many vertices will be degenerate. Here, we create a map of
      // the original vertices, and delete vertices that are degenerate. The polygons are rearranged so that they
      // reference unique vertices.
      EBGEOMETRY_ALWAYS_EXPECT(a_vertices.size() >= 3);
      EBGEOMETRY_ALWAYS_EXPECT(a_polygons.size() >= 1);

      // indexMap contains a map of the old-to-new indexing.
      std::vector<std::pair<Vec3, int>> vertexMap;
      std::map<int, int>                indexMap;

      // Create a map of the original vertices and sort it using a lexical comparison operator.
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

        // Insert vertex if it is not degenerate.
        if ((curVert - prevVert).length() > EBGeometry::Limits::min()) {
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
    EBGEOMETRY_ALWAYS_INLINE
    EBGeometry::DCEL::Mesh<MetaData>*
    readIntoDCEL(const std::string a_fileName) noexcept
    {
      const MeshParser::FileType fileType = MeshParser::getFileType(a_fileName);

      EBGEOMETRY_ALWAYS_EXPECT((fileType != MeshParser::FileType::Unsupported));

      // Build the mesh.
      EBGeometry::DCEL::Mesh<MetaData>* mesh = nullptr;

      switch (fileType) {
      case FileType::STL: {
        mesh = MeshParser::STL::readSingle<MetaData>(a_fileName);

        break;
      }
      case FileType::PLY: {
#if 1
#warning "EBGeometry_MeshParserImplem.hpp::readIntoDCEL - PLY code not implemented"
        EBGEOMETRY_ALWAYS_EXPECT(false);
#else
        mesh = MeshParser::PLY::readSingle<MetaData>(a_fileName);
#endif
      }
      case FileType::Unsupported: {
        std::cerr << "In file EBGeometry_MeshParserImplem.hpp: MeshParser::readIntoDCEL - unsupported file type\n";
      }
      }

      return mesh;
    }

    template <typename MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    EBGeometry::DCEL::Mesh<MetaData>*
    turnPolygonSoupIntoDCEL(const std::vector<EBGeometry::Vec3>& a_vertices,
                            const std::vector<std::vector<int>>& a_faces) noexcept
    {
      using namespace EBGeometry::DCEL;

      // Figure out the number of vertices, edges, and polygons.
      const int numVertices = a_vertices.size();
      const int numFaces    = a_faces.size();

      int numEdges = 0;
      for (const auto& polygon : a_faces) {
        numEdges += polygon.size();
      }

      // Allocate vertex, edge, and half edge lists.
      Vertex<MetaData>* vertices = new Vertex<MetaData>[numVertices];
      Edge<MetaData>*   edges    = new Edge<MetaData>[numEdges];
      Face<MetaData>*   faces    = new Face<MetaData>[numFaces];

      for (int v = 0; v < numVertices; v++) {
        vertices[v].setPosition(a_vertices[v]);
        vertices[v].setVertexList(vertices);
        vertices[v].setEdgeList(edges);
        vertices[v].setFaceList(faces);
      }

      for (int e = 0; e < numEdges; e++) {
        edges[e].setVertexList(vertices);
        edges[e].setEdgeList(edges);
        edges[e].setFaceList(faces);
      }

      for (int f = 0; f < numFaces; f++) {
        faces[f].setVertexList(vertices);
        faces[f].setEdgeList(edges);
        faces[f].setFaceList(faces);
      }

      // Build DCEL faces, edges, and vertices. In order to build the DCEL edge pairs we keep track of
      // all outgoing edges from each vertex.
      int edgeIndex = 0;

      std::map<int, std::vector<int>> outgoingEdgesMap;

      for (int faceIndex = 0; faceIndex < numFaces; faceIndex++) {
        EBGEOMETRY_ALWAYS_EXPECT(a_faces[faceIndex].size() >= 3);

        const auto& faceVertices    = a_faces[faceIndex];
        const auto  numFaceVertices = faceVertices.size();

        // Build the face -- it already has associated vertex, edge, and face lists so we only need
        // to associate the half edge.
        faces[faceIndex].setEdge(edgeIndex);

        // Now build the actual edges -- each edge must reference the origin vertex.
        const int firstEdge = edgeIndex;
        const int lastEdge  = firstEdge + numFaceVertices - 1;

        for (int v = 0; v < numFaceVertices; v++) {
          const int vertIndex = faceVertices[v];

          EBGEOMETRY_EXPECT(vertIndex >= 0);
          EBGEOMETRY_EXPECT(vertIndex < numVertices);
          EBGEOMETRY_EXPECT(edgeIndex >= 0);
          EBGEOMETRY_EXPECT(edgeIndex < numEdges);

          Edge<MetaData>&   curEdge = edges[edgeIndex];
          Vertex<MetaData>& curVert = vertices[vertIndex];

          curEdge.setFace(faceIndex);
          curEdge.setVertex(vertIndex);
          curVert.setEdge(edgeIndex);

          // Add a reference of the current face to the vertex-face map.
          outgoingEdgesMap[vertIndex].emplace_back(edgeIndex);

          // Move on to the vertex in the face list and define the next half edge.
          edgeIndex++;
        }

        // Fix up the indexing of next/previous edges for the newly defined edges -- After that, they have
        // everything except the pair edge (and internal things like normal vectors).
        edges[firstEdge].setPreviousEdge(lastEdge);
        edges[lastEdge].setNextEdge(firstEdge);

        for (int i = firstEdge; i <= lastEdge; i++) {
          if (i > firstEdge) {
            edges[i].setPreviousEdge(i - 1);
          }
          if (i < lastEdge) {
            edges[i].setNextEdge(i + 1);
          }
        }
      }

      EBGEOMETRY_EXPECT(edgeIndex == numEdges);

      // Associate pair edges - we've built a map of all the outgoing edges from each vertex, so
      // we iterate through that map and look for the other edges in the neighboring polygons that
      // share start/end vertices.
      for (const auto& m : outgoingEdgesMap) {
        const auto& vertex        = m.first;
        const auto& outgoingEdges = m.second;

        for (auto& curOutgoingEdgeIndex : outgoingEdges) {
          Edge<MetaData>& curOutgoingEdge = edges[curOutgoingEdgeIndex];

          EBGEOMETRY_EXPECT(curOutgoingEdgeIndex >= 0);
          EBGEOMETRY_EXPECT(curOutgoingEdgeIndex < numEdges);
          EBGEOMETRY_EXPECT(curOutgoingEdge.getVertex() == vertex);

          const int curVertexStart = curOutgoingEdge.getVertex();
          const int curVertexEnd   = curOutgoingEdge.getOtherVertex();

          EBGEOMETRY_EXPECT(curVertexStart >= 0);
          EBGEOMETRY_EXPECT(curVertexStart < numVertices);
          EBGEOMETRY_EXPECT(curVertexEnd >= 0);
          EBGEOMETRY_EXPECT(curVertexEnd < numVertices);

          // Now go through all the other outgoing edges, and look for an edge which points
          // opposite to this one.
          for (auto& otherOutgoingEdgeIndex : outgoingEdges) {
            EBGEOMETRY_EXPECT(otherOutgoingEdgeIndex >= 0);
            EBGEOMETRY_EXPECT(otherOutgoingEdgeIndex < numEdges);

            if (otherOutgoingEdgeIndex != curOutgoingEdgeIndex) {
              const int incomingEdgeIndex = edges[otherOutgoingEdgeIndex].getPreviousEdge();

              EBGEOMETRY_EXPECT(curOutgoingEdgeIndex != otherOutgoingEdgeIndex);
              EBGEOMETRY_EXPECT(incomingEdgeIndex >= 0);
              EBGEOMETRY_EXPECT(incomingEdgeIndex < numEdges);

              Edge<MetaData>& otherIncomingEdge = edges[incomingEdgeIndex];

              const int otherVertexStart = otherIncomingEdge.getVertex();
              const int otherVertexEnd   = otherIncomingEdge.getOtherVertex();

              EBGEOMETRY_EXPECT(otherVertexStart >= 0);
              EBGEOMETRY_EXPECT(otherVertexStart < numVertices);
              EBGEOMETRY_EXPECT(otherVertexEnd >= 0);
              EBGEOMETRY_EXPECT(otherVertexEnd < numVertices);

              if ((curVertexStart == otherVertexEnd) && (curVertexEnd == otherVertexStart)) {
                curOutgoingEdge.setPairEdge(incomingEdgeIndex);
                otherIncomingEdge.setPairEdge(curOutgoingEdgeIndex);
              }
            }
          }
        }
      }

      // Allocate a mesh. Then do a sanity check and reconcile the mesh, which will compute
      // internal parameters like normal vectors for the vertices, edges, and faces.
      Mesh<MetaData>* mesh = new Mesh<MetaData>(numVertices, numEdges, numFaces, vertices, edges, faces);

      mesh->sanityCheck();
      mesh->reconcile();

      return mesh;
    }

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
      case MeshParser::FileEncoding::Unknown: {
        std::cerr
          << "In file EBGeometry_MeshParser_STLImplem.hpp function STL::readMulti - unknown file encoding encountered\n";

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
        std::cerr << "EBGeometry_MeshParserImplem.hpp::MeshParser::STL::getEncoding -- could not open file '" +
                       a_fileName + "'\n";
      }

      return fileEncoding;
    }

    template <typename MetaData>
    EBGEOMETRY_INLINE
    std::vector<std::pair<EBGeometry::DCEL::Mesh<MetaData>*, std::string>>
    STL::readASCII(const std::string a_fileName) noexcept
    {
      std::vector<std::pair<EBGeometry::DCEL::Mesh<MetaData>*, std::string>> meshes;

      // Storage for full ASCII file and line numbers indicating where STL objects start/end
      std::vector<std::string>         fileContents;
      std::vector<std::pair<int, int>> firstLast;

      // Read the entire file and figure out where objects begin and end.
      std::ifstream filestream(a_fileName);
      if (filestream.is_open()) {
        std::string line;

        int curLine = 0;
        int first;
        int last;

        while (std::getline(filestream, line)) {
          fileContents.emplace_back(line);

          std::string       str;
          std::stringstream sstream(line);
          sstream >> str;

          if (str == "solid") {
            first = curLine;
          }
          else if (str == "endsolid") {
            last = curLine;

            firstLast.emplace_back(first, last);
          }

          curLine++;
        }
      }
      else {
        std::cerr << "EBGeometry_MeshParserImplem.hpp::MeshParser::STL::readASCII -- could not open file '" +
                       a_fileName + "'\n";
      }

      // Read STL objects into triangle soups and then turn the soup into DCEL meshes.
      for (const auto& fl : firstLast) {
        const int firstLine = fl.first;
        const int lastLine  = fl.second;

        // Read triangle soup and compress it.
        std::vector<Vec3>             vertices;
        std::vector<std::vector<int>> triangles;
        std::string                   objectName;

        MeshParser::STL::readSTLSoupASCII(vertices, triangles, objectName, fileContents, firstLine, lastLine);

        EBGEOMETRY_ALWAYS_EXPECT(!(MeshParser::containsDegeneratePolygons(vertices, triangles)));

        MeshParser::removeDegenerateVerticesFromSoup(vertices, triangles);

        const auto mesh = MeshParser::turnPolygonSoupIntoDCEL<MetaData>(vertices, triangles);

        meshes.emplace_back(mesh, objectName);
      }

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
                          std::vector<std::vector<int>>&  a_triangles,
                          std::string&                    a_objectName,
                          const std::vector<std::string>& a_fileContents,
                          const int                       a_firstLine,
                          const int                       a_lastLine) noexcept
    {
      // First line just holds the object name.
      std::stringstream ss(a_fileContents[a_firstLine]);

      std::string str;
      std::string str1;
      std::string str2;

      ss >> str1 >> str2;

      a_objectName = str2;

      std::vector<int>* curTriangle = nullptr;

      // Read facets and vertices.
      for (int line = a_firstLine + 1; line < a_lastLine; line++) {
        ss = std::stringstream(a_fileContents[line]);

        ss >> str;

        if (str == "facet") {
          a_triangles.emplace_back(std::vector<int>());

          curTriangle = &a_triangles.back();
        }
        else if (str == "vertex") {
          Real x;
          Real y;
          Real z;

          ss >> x >> y >> z;

          a_vertices.emplace_back(Vec3(x, y, z));
          curTriangle->emplace_back(a_vertices.size() - 1);

          // Throw an error if we end up creating more than 100 vertices -- in this case the 'endloop'
          // or 'endfacet' are missing
          if (curTriangle->size() > 100) {
            std::cerr << "EBGeometry_MeshParserImplem.hpp::MeshParser::STL::readSTLSoupASCII -- logic bust\n";

            break;
          }
        }
      }
    }
  } // namespace MeshParser
} // namespace EBGeometry

#endif
