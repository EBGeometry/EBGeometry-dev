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

    template <typename MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    bool
    containsDegeneratePolygons(const PolygonSoup<MetaData>& a_soup) noexcept
    {
      // TLDR: This routine runs through all the faces in the polygon soup, and checks if the vertex
      // coordinates are degenerate. Note that we check the POSITION of the vertex indices in the soup
      // since although the indices might be degenerate (e.g., a triangle with vertex indicies 0,1,2),
      // the vertex coordinates themselves could be degenerate (e.g., vertex indices 0 and 1 might refer
      // to the same physical position).

      bool hasDegenerateFaces = false;

      const auto& vertices = std::get<0>(a_soup);
      const auto& faces    = std::get<1>(a_soup);

      EBGEOMETRY_ALWAYS_EXPECT(vertices.size() >= 3);
      EBGEOMETRY_ALWAYS_EXPECT(faces.size() >= 1);

      for (const auto& face : faces) {
        const auto& faceVertices = face.first;

        std::vector<Vec3> vertexCoords;

        for (const auto& faceVertex : face.first) {
          vertexCoords.emplace_back(vertices[faceVertex]);
        }

        std::sort(vertexCoords.begin(), vertexCoords.end(), [](const Vec3& a, const Vec3& b) { return a.lessLX(b); });

        const int numVertexCoords = vertexCoords.size();

        EBGEOMETRY_ALWAYS_EXPECT(numVertexCoords >= 3);

        for (int i = 0; i < numVertexCoords; i++) {
          if (vertexCoords[i] == vertexCoords[(i + 1) % numVertexCoords]) {
            hasDegenerateFaces = true;
          }
        }
      }

      return hasDegenerateFaces;
    }

    template <typename MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    bool
    containsDegenerateVertices(const PolygonSoup<MetaData>& a_soup) noexcept
    {
      // TLDR: Polygon soups might contain degenerate vertices. For example, STL files define individual facets
      // rather than links between vertices, and in this case many vertices will be degenerate. This routine
      // checks if the vertex list in a soup contains such vertices.
      bool hasDegenerateVertices = false;

      const std::vector<Vec3>& vertices = std::get<0>(a_soup);

      EBGEOMETRY_ALWAYS_EXPECT(vertices.size() >= 3);

      std::vector<Vec3> sortedVertices;
      for (const auto& v : vertices) {
        sortedVertices.emplace_back(v);
      }

      std::sort(sortedVertices.begin(), sortedVertices.end());

      for (int i = 1; i < sortedVertices.size(); i++) {
        if (sortedVertices[i] == sortedVertices[i - 1]) {
          hasDegenerateVertices = true;
        }
      }

      return hasDegenerateVertices;
    }

    template <typename MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    void
    removeDegenerateVerticesFromSoup(PolygonSoup<MetaData>& a_soup) noexcept
    {
      // TLDR: Polygon soups might contain degenerate vertices. For example, STL files define individual facets
      // rather than links between vertices, and in this case many vertices will be degenerate. Here, we create
      // a map of the original vertices, and delete vertices that are degenerate. The polygons are rearranged
      // so that they reference unique vertices.

      std::vector<Vec3>&                                  vertices = std::get<0>(a_soup);
      std::vector<std::pair<std::vector<int>, MetaData>>& faces    = std::get<1>(a_soup);

      EBGEOMETRY_ALWAYS_EXPECT(vertices.size() >= 3);
      EBGEOMETRY_ALWAYS_EXPECT(faces.size() >= 1);

      // vertexMap contains the original vertices together with their original index.
      // indexMap contains a map of the old-to-new indexing, and is built as we traverse
      // the vertex list.
      std::vector<std::pair<Vec3, int>> vertexMap;
      std::map<int, int>                indexMap;

      // Create a map of the original vertices and sort it using a lexical comparison operator.
      for (int i = 0; i < vertices.size(); i++) {
        vertexMap.emplace_back(vertices[i], i);
      }

      std::sort(vertexMap.begin(), vertexMap.end(), [](const std::pair<Vec3, int>& A, const std::pair<Vec3, int>& B) {
        const Vec3& a = A.first;
        const Vec3& b = B.first;

        return a.lessLX(b);
      });

      // Toss the original vertex coordinates and begin anew.
      vertices.resize(0);

      for (int i = 0; i < vertexMap.size(); i++) {
        const Vec3 originalCoord = vertexMap[i].first;
        const int  originalIndex = vertexMap[i].second;

        // Only insert a vertex in the vertex vector if it's position is different from
        // the previous entry in the vertex map. The original index will then be mapped
        // to a new index in the new vertex vector.
        if (i == 0) {
          vertices.emplace_back(originalCoord);
        }
        else {
          const Vec3 previousCoord = vertexMap[i - 1].first;
          if ((originalCoord - previousCoord).length() > EBGeometry::Limits::min()) {
            vertices.emplace_back(originalCoord);
          }
        }

        indexMap.emplace(originalIndex, vertices.size() - 1);
      }

      // Fix up the polygon indexing which should reference the non-degenerate vertices.
      for (int i = 0; i < faces.size(); i++) {
        std::vector<int>& faceIndices = faces[i].first;

        EBGEOMETRY_EXPECT(faceIndices.size() >= 3);

        for (int v = 0; v < faceIndices.size(); v++) {
          EBGEOMETRY_EXPECT(faceIndices[v] >= 0);

          faceIndices[v] = indexMap.at(faceIndices[v]);
        }
      }
    }

    template <typename MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    PolygonSoup<MetaData>
    readIntoSoup(const std::string a_fileName) noexcept
    {
      PolygonSoup<MetaData> soup;

      switch (MeshParser::getFileType(a_fileName)) {
      case FileType::STL: {

        // The extra assertions are here because while I normally expect that STL files contain a single object,
        // there is nothing in the STL standard that requires it. Eventually, someone might require us to read
        // multiple STL files in one go, so I'm letting all the parsers read whatever is in the file, and then
        // we (for now) only use the first object.
        const auto soups = MeshParser::STL::readIntoPolygonSoup<MetaData>(a_fileName);

        EBGEOMETRY_ALWAYS_EXPECT(soups.size() == 1);

        soup = soups[0];

        break;
      }
      case FileType::PLY: {
        const auto soups = MeshParser::PLY::readIntoPolygonSoup<MetaData>(a_fileName);

        EBGEOMETRY_ALWAYS_EXPECT(soups.size() == 1);

        soup = soups[0];

        break;
      }
      case FileType::Unsupported: {
        std::cerr << "In file EBGeometry_MeshParserImplem.hpp::readIntoSoup - unsupported file type\n";

        break;
      }
      }

      return soup;
    }

    template <typename MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    EBGeometry::DCEL::Mesh<MetaData>*
    readIntoDCEL(const std::string a_fileName) noexcept
    {
      // Read file into a polygon soup and remove degenerate vertices.
      PolygonSoup<MetaData> soup = MeshParser::readIntoSoup<MetaData>(a_fileName);

      MeshParser::removeDegenerateVerticesFromSoup(soup);

      EBGEOMETRY_EXPECT(!(MeshParser::containsDegeneratePolygons(soup)));
      EBGEOMETRY_EXPECT(!(MeshParser::containsDegenerateVertices(soup)));

      const auto mesh = MeshParser::turnPolygonSoupIntoDCEL<MetaData>(soup);

      return mesh;
    }

    template <typename MetaData>
    EBGEOMETRY_ALWAYS_INLINE
    EBGeometry::DCEL::Mesh<MetaData>*
    turnPolygonSoupIntoDCEL(PolygonSoup<MetaData>& a_soup) noexcept
    {
      using namespace EBGeometry::DCEL;

      EBGEOMETRY_EXPECT(!(MeshParser::containsDegeneratePolygons(a_soup)));
      EBGEOMETRY_EXPECT(!(MeshParser::containsDegenerateVertices(a_soup)));

      std::vector<Vec3>&                                  soupVertices = std::get<0>(a_soup);
      std::vector<std::pair<std::vector<int>, MetaData>>& soupFaces    = std::get<1>(a_soup);

      EBGEOMETRY_ALWAYS_EXPECT(soupVertices.size() >= 3);
      EBGEOMETRY_ALWAYS_EXPECT(soupFaces.size() >= 1);

      const int numVertices = soupVertices.size();
      const int numFaces    = soupFaces.size();

      int numEdges = 0;
      for (const auto& f : soupFaces) {
        numEdges += (f.first).size();
      }

      // Allocate vertex, edge, and half edge lists.
      Vertex<MetaData>* vertices = new Vertex<MetaData>[numVertices];
      Edge<MetaData>*   edges    = new Edge<MetaData>[numEdges];
      Face<MetaData>*   faces    = new Face<MetaData>[numFaces];

      for (int v = 0; v < numVertices; v++) {
        vertices[v].setPosition(soupVertices[v]);
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
        const auto& faceVertices    = soupFaces[faceIndex].first;
        const auto& faceMetaData    = soupFaces[faceIndex].second;
        const auto  numFaceVertices = faceVertices.size();

        EBGEOMETRY_ALWAYS_EXPECT(faceVertices.size() >= 3);

        // Build the face -- it already has associated vertex, edge, and face lists so we only need
        // to associate the half edge.
        faces[faceIndex].setEdge(edgeIndex);
        faces[faceIndex].setMetaData(faceMetaData);

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

      const bool isManifold = mesh->isManifold();

      if (isManifold) {
        mesh->reconcile();
      }
      else {
        EBGEOMETRY_ALWAYS_EXPECT(isManifold);

        mesh->freeMem();

        delete mesh;

        mesh = nullptr;

        std::cerr
          << "EBGeometry_MeshParserImplem.hpp::turnPolygonSoupIntoDCEL - mesh is not manifold and will not be turned into a DCEL mesh!\n";
      }

      return mesh;
    }

    template <typename MetaData>
    EBGEOMETRY_INLINE
    std::vector<PolygonSoup<MetaData>>
    STL::readIntoPolygonSoup(const std::string a_fileName) noexcept
    {
      const auto fileEncoding = MeshParser::STL::getEncoding(a_fileName);

      EBGEOMETRY_ALWAYS_EXPECT(MeshParser::getFileType(a_fileName) == MeshParser::FileType::STL);
      EBGEOMETRY_ALWAYS_EXPECT(fileEncoding == MeshParser::FileEncoding::ASCII ||
                               fileEncoding == MeshParser::FileEncoding::Binary);

      std::vector<PolygonSoup<MetaData>> soups;

      switch (fileEncoding) {
      case MeshParser::FileEncoding::ASCII: {
        soups = EBGeometry::MeshParser::STL::readASCII<MetaData>(a_fileName);

        break;
      }
      case MeshParser::FileEncoding::Binary: {
        soups = EBGeometry::MeshParser::STL::readBinary<MetaData>(a_fileName);

        break;
      }
      case MeshParser::FileEncoding::Unknown: {
        std::cerr << "EBGeometry_MeshParserImplem.hpp::STL::readIntoPolygonSoup - unknown file encoding encountered\n";

        break;
      }
      }

      return soups;
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
        std::cerr << "EBGeometry_MeshParserImplem.hpp::STL::getEncoding -- could not open file '" + a_fileName + "'\n";
      }

      return fileEncoding;
    }

    template <typename MetaData>
    EBGEOMETRY_INLINE
    std::vector<PolygonSoup<MetaData>>
    STL::readBinary(const std::string a_fileName) noexcept
    {
      EBGEOMETRY_ALWAYS_EXPECT(MeshParser::getFileType(a_fileName) == MeshParser::FileType::STL);
      EBGEOMETRY_ALWAYS_EXPECT(MeshParser::STL::getEncoding(a_fileName) == MeshParser::FileEncoding::Binary);

      std::vector<PolygonSoup<MetaData>> soups;

      using VertexList   = std::vector<Vec3>;
      using TriangleList = std::vector<std::pair<std::vector<int>, MetaData>>;

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

          objectFacets.emplace_back(std::make_pair(curFacet, MetaData()));
        }

        // Turn the triangle soup into a mesh.
        for (const auto& soup : verticesAndTriangles) {
          const auto& id        = soup.first;
          const auto& vertices  = soup.second.first;
          const auto& triangles = soup.second.second;
          const auto  stringID  = a_fileName + "_" + std::to_string(id);

          soups.emplace_back(std::make_tuple(vertices, triangles, stringID));
        }
      }
      else {
        std::cerr << "EBGeometry::MeshParser::STL::readBinary -- could not open file '" + a_fileName + "'\n";
      }

      return soups;
    }

    template <typename MetaData>
    EBGEOMETRY_INLINE
    std::vector<PolygonSoup<MetaData>>
    STL::readASCII(const std::string a_fileName) noexcept
    {
      EBGEOMETRY_ALWAYS_EXPECT(MeshParser::getFileType(a_fileName) == MeshParser::FileType::STL);
      EBGEOMETRY_ALWAYS_EXPECT(MeshParser::STL::getEncoding(a_fileName) == MeshParser::FileEncoding::ASCII);

      std::vector<PolygonSoup<MetaData>> soups;

      // Storage for full ASCII file and line numbers indicating where STL objects start/end
      std::vector<std::string>         fileContents;
      std::vector<std::pair<int, int>> firstLast;

      // Read the entire file and figure out where objects begin and end -- have to do this in case there are multiple objects in
      // the same STL file.
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
        std::cerr << "EBGeometry_MeshParserImplem.hpp::STL::readASCII -- could not open file '" + a_fileName + "'\n";
      }

      // Read STL objects into triangle soups and then turn the soup into DCEL meshes.
      for (const auto& fl : firstLast) {
        const int firstLine = fl.first;
        const int lastLine  = fl.second;

        soups.emplace_back(PolygonSoup<MetaData>());

        auto& vertices   = std::get<0>(soups.back());
        auto& triangles  = std::get<1>(soups.back());
        auto& objectName = std::get<2>(soups.back());

        // First line just holds the object name.
        std::stringstream ss(fileContents[firstLine]);

        std::string str;
        std::string str1;
        std::string str2;

        ss >> str1 >> str2;

        objectName = a_fileName + "_" + str2;

        // Read facets and vertices.
        for (int line = firstLine + 1; line < lastLine; line++) {
          ss = std::stringstream(fileContents[line]);

          ss >> str;

          if (str == "facet") {
            triangles.emplace_back(std::make_pair(std::vector<int>(), MetaData()));
          }
          else if (str == "vertex") {
            Real x;
            Real y;
            Real z;

            ss >> x >> y >> z;

            vertices.emplace_back(Vec3(x, y, z));

            auto& curTriangle = (triangles.back()).first;

            curTriangle.emplace_back(vertices.size() - 1);

            EBGEOMETRY_ALWAYS_EXPECT(curTriangle.size() < 100);
          }
        }
      }

      return soups;
    }

    template <typename MetaData>
    EBGEOMETRY_INLINE
    std::vector<PolygonSoup<MetaData>>
    PLY::readIntoPolygonSoup(const std::string a_fileName) noexcept
    {
      const auto fileEncoding = MeshParser::PLY::getEncoding(a_fileName);

      EBGEOMETRY_ALWAYS_EXPECT(MeshParser::getFileType(a_fileName) == MeshParser::FileType::PLY);
      EBGEOMETRY_ALWAYS_EXPECT(fileEncoding == MeshParser::FileEncoding::ASCII ||
                               fileEncoding == MeshParser::FileEncoding::Binary);

      std::vector<PolygonSoup<MetaData>> soups;

      switch (fileEncoding) {
      case MeshParser::FileEncoding::ASCII: {
        soups = EBGeometry::MeshParser::PLY::readASCII<MetaData>(a_fileName);

        break;
      }
      case MeshParser::FileEncoding::Binary: {
        soups = EBGeometry::MeshParser::PLY::readBinary<MetaData>(a_fileName);

        break;
      }
      case MeshParser::FileEncoding::Unknown: {
        std::cerr << "EBGeometry_MeshParserImplem.hpp::PLY::readIntoPolygonSoup - unknown file encoding encountered\n";

        break;
      }
      }

      return soups;
    }

    EBGEOMETRY_INLINE
    MeshParser::FileEncoding
    PLY::getEncoding(const std::string a_fileName) noexcept
    {
      MeshParser::FileEncoding fileEncoding = MeshParser::FileEncoding::Unknown;

      std::ifstream is(a_fileName, std::istringstream::in | std::ios::binary);
      if (is.good()) {

        std::string line;
        std::string str1;
        std::string str2;

        // Ignore first line.
        std::getline(is, line);
        std::getline(is, line);

        std::stringstream ss(line);

        ss >> str1 >> str2;

        if (str2 == "ascii") {
          fileEncoding = MeshParser::FileEncoding::ASCII;
        }
        else if (str2 == "binary_little_endian") {
          fileEncoding = MeshParser::FileEncoding::Binary;
        }
        else if (str2 == "binary_big_endian") {
          fileEncoding = MeshParser::FileEncoding::Binary;
        }
      }
      else {
        std::cerr << "MeshParser::PLY::getEncoding -- could not open file '" + a_fileName + "'\n";
      }

      return fileEncoding;
    }

    template <typename MetaData>
    EBGEOMETRY_INLINE
    std::vector<PolygonSoup<MetaData>>
    PLY::readASCII(const std::string a_fileName) noexcept
    {
      EBGEOMETRY_ALWAYS_EXPECT(MeshParser::getFileType(a_fileName) == MeshParser::FileType::PLY);
      EBGEOMETRY_ALWAYS_EXPECT(MeshParser::PLY::getEncoding(a_fileName) == MeshParser::FileEncoding::ASCII);

      std::vector<PolygonSoup<MetaData>> soups;

      std::ifstream filestream(a_fileName);
      if (filestream.is_open()) {

        Real x;
        Real y;
        Real z;

        int numVertices;
        int numFaces;
        int numProcessed;
        int numVerticesInPolygon;

        std::string str1;
        std::string str2;
        std::string line;
      }
      else {
        std::cerr << "EBGeometry_MeshParserImplem.hpp::PLY::readASCII - could not open file '" + a_fileName + "'\n";
      }

      return soups;
    }

    template <typename MetaData>
    EBGEOMETRY_INLINE
    std::vector<PolygonSoup<MetaData>>
    PLY::readBinary(const std::string a_fileName) noexcept
    {
      std::vector<PolygonSoup<MetaData>> soups;

#warning "EBGeometry_MeshParserImplem.hpp::PLY::readBinary - not implemented"

      return soups;
    }
  }   // namespace MeshParser
  }   // namespace MeshParser

#endif
