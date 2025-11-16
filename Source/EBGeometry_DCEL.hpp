// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   EBGeometry_DCEL.hpp
 * @brief  DCEL namespace with forward declarations.
 * @author Robert Marskar
 */

#ifndef EBGEOMETRY_DCEL_HPP
#define EBGEOMETRY_DCEL_HPP

/**
 * @namespace EBGeometry::DCEL
 * @brief Doubly-Connected Edge List (DCEL) data structure for topological mesh representation.
 *
 * @details The DCEL namespace provides a half-edge data structure for representing and querying
 *          watertight surface meshes. DCEL is a topological mesh representation that explicitly
 *          stores connectivity information between vertices, edges, and faces, enabling efficient
 *          traversal and geometric queries.
 *
 * ## DCEL Structure
 *
 * A DCEL mesh consists of three main components:
 * - **Vertices** (Vertex): Points in 3D space with references to incident edges
 * - **Half-Edges** (Edge): Directed edges forming the boundary of faces, with references to:
 *   - Starting vertex
 *   - Twin edge (opposite direction)
 *   - Next edge (counterclockwise around face)
 *   - Previous edge
 *   - Incident face
 * - **Faces** (Face): Polygonal faces with references to bounding half-edges
 *
 * ## Key Features
 *
 * - **Topological Consistency**: The half-edge structure ensures that every edge has a twin,
 *   maintaining manifold connectivity for watertight meshes.
 * - **Efficient Traversal**: Constant-time access to neighboring elements (vertices, edges, faces).
 * - **Signed Distance Queries**: Built-in support for computing signed distances to mesh surfaces.
 * - **Templated Metadata**: Each primitive (vertex, edge, face) can store user-defined metadata
 *   of type `Meta` (default: `DefaultMetaData = short`).
 * - **Normal Computation**: Automatic computation of vertex and edge normals for smooth surface
 *   representation.
 *
 * ## Usage
 *
 * DCEL meshes are typically constructed by parsers (e.g., `MeshParser`) that read geometry from
 * files (STL, PLY, OBJ) and build the topological structure. The mesh is then often embedded in
 * a bounding volume hierarchy (BVH) for accelerated spatial queries.
 *
 * Example workflow:
 * @code
 * // 1. Parse mesh from file
 * MeshParser<short> parser("mesh.stl");
 * auto mesh = parser.getMesh();
 *
 * // 2. Use mesh directly (slow for large meshes)
 * Real distance = mesh->value(query_point);
 *
 * // 3. Or embed in BVH for fast queries
 * BVH<DCEL::Face<short>> bvh(mesh);
 * Real fast_distance = bvh.value(query_point);
 * @endcode
 *
 * ## Performance Considerations
 *
 * - **Direct Queries**: The DCEL mesh supports direct signed distance queries by iterating over
 *   all faces. This is O(N) and inefficient for large meshes.
 * - **BVH Acceleration**: For production use, DCEL meshes should be wrapped in a BVH to achieve
 *   O(log N) query performance.
 *
 * ## Memory Layout
 *
 * The Mesh class itself is non-owning—it stores pointers to externally managed arrays of vertices,
 * edges, and faces. This allows flexible memory management and GPU compatibility.
 *
 * @see Vertex for vertex data structure and operations
 * @see Edge for half-edge data structure and operations
 * @see Face for face data structure and operations
 * @see Mesh for complete mesh representation with signed distance queries
 *
 * @note This is an advanced data structure. For simple triangle soups without topological queries,
 *       consider using TriangleCollection instead.
 */
namespace EBGeometry::DCEL {

  /**
   * @brief Weighting scheme for vertex normal computation.
   * @details Determines how face normals are combined when computing vertex normals
   *          at shared vertices. Different weighting schemes affect the smoothness
   *          of the resulting surface representation.
   */
  enum class VertexNormalWeight
  {
    /**
     * @brief Uniform weighting (simple average).
     * @details All incident face normals contribute equally to the vertex normal,
     *          regardless of face size or angle. Fastest to compute but may produce
     *          poor results for irregular meshes.
     */
    None,

    /**
     * @brief Angle-weighted averaging.
     * @details Each face normal is weighted by the angle it subtends at the vertex.
     *          This produces smoother, more accurate normals and is recommended for
     *          most use cases. Slightly more expensive than uniform weighting.
     */
    Angle
  };

  /**
   * @brief Default metadata type for DCEL primitives.
   * @details The metadata type is used to store user-defined data on vertices, edges, and faces.
   *          Common use cases include storing face indices, material IDs, or other application-specific
   *          information. The default type is `short` to minimize memory overhead.
   */
  using DefaultMetaData = short;

  /**
   * @brief DCEL vertex class.
   * @details Represents a vertex in the DCEL mesh, storing position, normal, and references
   *          to incident half-edges for topological traversal.
   *
   * @tparam Meta User-defined metadata type (default: DefaultMetaData).
   *
   * @see EBGeometry_DCEL_Vertex.hpp for full declaration
   */
  template <class Meta = DefaultMetaData>
  class Vertex;

  /**
   * @brief DCEL half-edge class.
   * @details Represents a directed half-edge in the DCEL mesh. Each geometric edge is represented
   *          by two half-edges pointing in opposite directions (twins). Half-edges store references
   *          to their starting vertex, twin, next/previous edges, and incident face.
   *
   * @tparam Meta User-defined metadata type (default: DefaultMetaData).
   *
   * @see EBGeometry_DCEL_Edge.hpp for full declaration
   */
  template <class Meta = DefaultMetaData>
  class Edge;

  /**
   * @brief DCEL face class.
   * @details Represents a polygonal face in the DCEL mesh, storing a reference to one of its
   *          bounding half-edges. Faces support signed distance queries and normal computation.
   *
   * @tparam Meta User-defined metadata type (default: DefaultMetaData).
   *
   * @see EBGeometry_DCEL_Face.hpp for full declaration
   */
  template <class Meta = DefaultMetaData>
  class Face;

  /**
   * @brief DCEL mesh class.
   * @details Complete DCEL mesh representation with non-owning pointers to vertex, edge, and face
   *          arrays. Provides signed distance function queries and topological operations.
   *          Typically constructed by parsers (e.g., MeshParser) from geometry files.
   *
   * @tparam Meta User-defined metadata type (default: DefaultMetaData).
   *
   * @see EBGeometry_DCEL_Mesh.hpp for full declaration
   *
   * @warning This class is complex and should not be constructed manually. Use file parsers
   *          (STL, PLY, OBJ) to build DCEL meshes from geometry data.
   */
  template <class Meta = DefaultMetaData>
  class Mesh;
} // namespace EBGeometry::DCEL

#endif
