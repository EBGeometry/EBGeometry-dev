// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   EBGeometry_DCEL_Vertex.hpp
 * @brief  Declaration of a vertex class for use in DCEL descriptions of polygon
 * tessellations.
 * @author Robert Marskar
 *
 * @details This file provides the Vertex class for doubly-connected edge list (DCEL)
 * mesh representations. The implementation uses EBGeometry::Span for non-owning
 * references to mesh data, ensuring GPU compatibility and memory safety.
 *
 * Key features:
 * - Trivially copyable and standard layout for GPU execution
 * - Robust normal computation algorithms with numerical safeguards
 * - Comprehensive bounds checking to prevent undefined behavior
 * - Support for both simple averaging and angle-weighted normal computation
 */

#ifndef EBGEOMETRY_DCEL_VERTEX_HPP
#define EBGEOMETRY_DCEL_VERTEX_HPP

// Our includes
#include "EBGeometry_DCEL.hpp"
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_GPUTypes.hpp"
#include "EBGeometry_Macros.hpp"
#include "EBGeometry_Span.hpp"
#include "EBGeometry_Vec.hpp"

namespace EBGeometry::DCEL {

  /**
   * @brief Class which represents a vertex node in a doubly-connected edge list (DCEL).
   * @details This class is used in DCEL functionality which stores polygonal
   * surfaces in a mesh. The Vertex class has a position, a normal vector, and a
   * reference (index) to one of the outgoing edges from the vertex.
   *
   * The vertex stores non-owning Span references to the complete mesh arrays
   * (vertices, edges, faces), enabling efficient topological traversal and normal
   * computation algorithms.
   *
   * @tparam MetaData User-defined metadata type attached to the vertex.
   *
   * @note The normal vector is outgoing, i.e. a point x is "outside" the vertex if
   * the dot product between n and (x - x0) is positive.
   *
   * @note This class is trivially copyable and has standard layout, making it
   * suitable for GPU execution. The Span references can be safely copied to GPU
   * memory and used in device code.
   *
   * @note Normal computation algorithms (computeVertexNormalAverage and
   * computeVertexNormalAngleWeighted) include robust bounds checking and
   * numerical safeguards to prevent undefined behavior.
   */
  template <class MetaData>
  class Vertex
  {
  public:
    /**
     * @brief Empty constructor.
     * @details This initializes the position and the normal vector to zero
     * vectors, and the polygon face list is empty
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Vertex() noexcept = default;

    /**
     * @brief Partial constructor.
     * @param[in] a_position Vertex position
     * @details This initializes the position to a_position and the normal vector
     * to the zero vector. The polygon face list is empty.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Vertex(const Vec3& a_position) noexcept;

    /**
     * @brief Constructor.
     * @param[in] a_position Vertex position
     * @param[in] a_normal Vertex normal vector
     * @details This initializes the position to a_position and the normal vector
     * to a_normal. The polygon face list is empty.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Vertex(const Vec3& a_position, const Vec3& a_normal) noexcept;

    /**
     * @brief Full constructor.
     * @param[in] a_position Vertex position
     * @param[in] a_normal Vertex normal vector
     * @param[in] a_edge Outgoing half-edge index
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Vertex(const Vec3& a_position, const Vec3& a_normal, int a_edge) noexcept;

    /**
     * @brief Copy constructor
     * @param[in] a_otherVertex Other vertex
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Vertex(const Vertex& a_otherVertex) noexcept = default;

    /**
     * @brief Move constructor
     * @param[in, out] a_otherVertex Other vertex
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Vertex(Vertex&& a_otherVertex) noexcept = default;

    /**
     * @brief Destructor
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    ~Vertex() noexcept = default;

    /**
     * @brief Copy assignment
     * @param[in] a_vertex Other vertex
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Vertex&
    operator=(const Vertex& a_vertex) noexcept = default;

    /**
     * @brief Move assignment
     * @param[in] a_vertex Other vertex
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Vertex&
    operator=(Vertex&& a_vertex) noexcept = default;

    /**
     * @brief Set the vertex position
     * @param[in] a_position Vertex position
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    setPosition(const Vec3& a_position) noexcept;

    /**
     * @brief Set the vertex normal vector
     * @param[in] a_normal Vertex normal vector
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    setNormal(const Vec3& a_normal) noexcept;

    /**
     * @brief Set the outgoing edge.
     * @param[in] a_edge Outgoing half-edge index
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    setEdge(int a_edge) noexcept;

    /**
     * @brief Set the metadata
     * @param[in] a_metaData
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    setMetaData(const MetaData& a_metaData) noexcept;

    /**
     * @brief Set the vertex list.
     * @param[in] a_vertexList Span view of vertices
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    setVertexList(EBGeometry::Span<const Vertex<MetaData>> a_vertexList) noexcept;

    /**
     * @brief Set the edge list.
     * @param[in] a_edgeList Span view of edges
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    setEdgeList(EBGeometry::Span<const Edge<MetaData>> a_edgeList) noexcept;

    /**
     * @brief Set the face list.
     * @param[in] a_faceList Span view of faces
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    setFaceList(EBGeometry::Span<const Face<MetaData>> a_faceList) noexcept;

    /**
     * @brief Get the vertex list
     * @return m_vertexList
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    EBGeometry::Span<const Vertex<MetaData>>
    getVertexList() const noexcept;

    /**
     * @brief Get the edge list
     * @return m_edgeList
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    EBGeometry::Span<const Edge<MetaData>>
    getEdgeList() const noexcept;

    /**
     * @brief Get the face list.
     * @return m_faceList
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    EBGeometry::Span<const Face<MetaData>>
    getFaceList() const noexcept;

    /**
     * @brief Normalize the normal vector to a length of 1.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    normalizeNormalVector() noexcept;

    /**
     * @brief Compute the vertex normal, using an average of the normal vectors of all faces
     * sharing this vertex.
     * @details This computes the vertex normal as n = sum(normal(face))/num(faces).
     *
     * The algorithm traverses the half-edge cycle around this vertex, accumulating
     * face normals. The implementation includes:
     * - Bounds checking on all array accesses
     * - Validation of Span data pointers
     * - Infinite loop detection for malformed DCEL structures
     *
     * @note Requires valid mesh topology (properly connected half-edges forming a cycle).
     * @note The outgoing edge index (m_outgoingEdge) must be set before calling.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    computeVertexNormalAverage() noexcept;

    /**
     * @brief Compute the vertex normal, using the pseudonormal algorithm which
     * weights the normal with the subtended angle to each connected face.
     * @details This computes the normal vector using the pseudonormal algorithm from
     * Baerentzen and Aanes in "Signed distance computation using the angle
     * weighted pseudonormal" (DOI: 10.1109/TVCG.2005.49).
     *
     * The algorithm weights each face normal by the angle it subtends at this vertex:
     * n = sum(angle_i * normal_i) / |sum(angle_i * normal_i)|
     *
     * The implementation includes robust numerical handling:
     * - Bounds checking on all array accesses
     * - Division by zero prevention for edge normalization
     * - Domain clamping for acos to prevent NaN from floating-point errors
     * - Infinite loop detection for malformed DCEL structures
     * - Validation of non-degenerate geometry
     *
     * @note Requires valid mesh topology with non-degenerate faces.
     * @note More accurate than simple averaging for irregular meshes.
     * @note The outgoing edge index (m_outgoingEdge) must be set before calling.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    computeVertexNormalAngleWeighted() noexcept;

    /**
     * @brief Get the outgoing edge
     * @return m_edge
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    int
    getEdge() const noexcept;

    /**
     * @brief Return modifiable vertex position.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Vec3&
    getPosition() noexcept;

    /**
     * @brief Return immutable vertex position.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    const Vec3&
    getPosition() const noexcept;

    /**
     * @brief Return modifiable vertex normal vector.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Vec3&
    getNormal() noexcept;

    /**
     * @brief Return immutable vertex normal vector.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    const Vec3&
    getNormal() const noexcept;

    /**
     * @brief Return outgoing edge index
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    int
    getOutgoingEdge() const noexcept;

    /**
     * @brief Get the signed distance to this vertex
     * @param[in] a_x0 Position in space.
     * @return The returned distance is |a_x0 - m_position| and the sign is given
     * by the sign of m_normal * |a_x0 - m_position|.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Real
    signedDistance(const Vec3& a_x0) const noexcept;

    /**
     * @brief Get the squared unsigned distance to this vertex
     * @details This is faster to compute than signedDistance, and might be
     * preferred for some algorithms.
     * @return Returns the vector length of (a_x - m_position)
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Real
    unsignedDistance2(const Vec3& a_x0) const noexcept;

    /**
     * @brief Get meta-data
     * @return m_metaData
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    MetaData&
    getMetaData() noexcept;

    /**
     * @brief Get immutable meta-data
     * @return m_metaData
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    const MetaData&
    getMetaData() const noexcept;

  protected:
    /**
     * @brief Span view of all vertices in the mesh.
     */
    EBGeometry::Span<const Vertex<MetaData>> m_vertexList{};

    /**
     * @brief Span view of all half-edges in the mesh.
     */
    EBGeometry::Span<const Edge<MetaData>> m_edgeList{};

    /**
     * @brief Span view of all faces in the mesh.
     */
    EBGeometry::Span<const Face<MetaData>> m_faceList{};

    /**
     * @brief Outgoing edge.
     */
    int m_outgoingEdge = -1;

    /**
     * @brief Vertex position
     */
    Vec3 m_position = Vec3::zero();

    /**
     * @brief Vertex normal vector
     */
    Vec3 m_normal = Vec3::zero();

    /**
     * @brief MetaData-data for this vertex
     */
    MetaData m_metaData = MetaData();
  };

  // Static assertions to ensure GPU compatibility
  static_assert(std::is_trivially_copyable_v<Vertex<DefaultMetaData>>,
                "EBGeometry::DCEL::Vertex must be trivially copyable for GPU compatibility");
  static_assert(std::is_standard_layout_v<Vertex<DefaultMetaData>>,
                "EBGeometry::DCEL::Vertex must have standard layout for GPU compatibility");

} // namespace EBGeometry::DCEL

#include "EBGeometry_DCEL_VertexImplem.hpp" // NOLINT

#endif
