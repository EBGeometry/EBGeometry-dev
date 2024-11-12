/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_DCEL_Vertex.hpp
  @brief  Declaration of a vertex class for use in DCEL descriptions of polygon
  tessellations.
  @author Robert Marskar
*/

#ifndef EBGeometry_DCEL_Vertex
#define EBGeometry_DCEL_Vertex

// Our includes
#include "EBGeometry_DCEL.hpp"
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_GPUTypes.hpp"
#include "EBGeometry_Macros.hpp"
#include "EBGeometry_Vec.hpp"

namespace EBGeometry::DCEL {

  /*!
    @brief Class which represents a vertex node in a double-edge connected list (DCEL).
    @details This class is used in DCEL functionality which stores polygonal
    surfaces in a mesh. The Vertex class has a position, a normal vector, and a
    reference (index) to one of the outgoing edges from the vertex.
      
    @note The normal vector is outgoing, i.e. a point x is "outside" the vertex if
    the dot product between n and (x - x0) is positive.

    @note This class is GPU-copyable with the exception of the edge list which must be
    set appropriately for the device storage.
  */
  template <class MetaData>
  class Vertex
  {
  public:
    /*!
      @brief Empty constructor.
      @details This initializes the position and the normal vector to zero
      vectors, and the polygon face list is empty
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Vertex() noexcept;

    /*!
      @brief Partial constructor.
      @param[in] a_position Vertex position
      @details This initializes the position to a_position and the normal vector
      to the zero vector. The polygon face list is empty.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Vertex(const Vec3& a_position) noexcept;

    /*!
      @brief Constructor.
      @param[in] a_position Vertex position
      @param[in] a_normal Vertex normal vector
      @details This initializes the position to a_position and the normal vector
      to a_normal. The polygon face list is empty.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Vertex(const Vec3& a_position, const Vec3& a_normal) noexcept;

    /*!
      @brief Full constructor.
      @param[in] a_position Vertex position
      @param[in] a_normal Vertex normal vector
      @param[in] a_edge Outgoing half-edge index
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Vertex(const Vec3& a_position, const Vec3& a_normal, int a_edge) noexcept;

    /*!
      @brief Copy constructor
      @param[in] a_otherVertex Other vertex
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Vertex(const Vertex& a_otherVertex) noexcept = default;

    /*!
      @brief Move constructor
      @param[in, out] a_otherVertex Other vertex
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Vertex(Vertex&& a_otherVertex) noexcept = default;

    /*!
      @brief Destructor
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    ~Vertex() noexcept = default;

    /*!
      @brief Copy assignment
      @param[in] a_vertex Other vertex
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Vertex&
    operator=(const Vertex& a_vertex) noexcept = default;

    /*!
      @brief Move assignment
      @param[in] a_vertex Other vertex
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Vertex&
    operator=(Vertex&& a_vertex) noexcept = default;

    /*!
      @brief Define function
      @param[in] a_position Vertex position
      @param[in] a_normal   Vertex normal vector
      @param[in] a_edge     Outgoing half-edge index
      @details This sets the position, normal vector, and half-edge index
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    define(const Vec3& a_position, const Vec3& a_normal, int a_edge) noexcept;

    /*!
      @brief Set the vertex position
      @param[in] a_position Vertex position
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    setPosition(const Vec3& a_position) noexcept;

    /*!
      @brief Set the vertex normal vector
      @param[in] a_normal Vertex normal vector
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    setNormal(const Vec3& a_normal) noexcept;

    /*!
      @brief Set the outgoing edge.
      @param[in] a_edge Outgoing half-edge index
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    setEdge(int a_edge) noexcept;

    /*!
      @brief Set the metadata
      @param[in] a_metaData
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    setMetaData(const MetaData& a_metaData) noexcept;

    /*!
      @brief Set the vertex list.
      @param[in] a_vertexList List (malloc'ed array) of vertices
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    setVertexList(const Vertex<MetaData>* a_vertexList) noexcept;

    /*!
      @brief Set the edge list.
      @param[in] a_edgeList List (malloc'ed array) of edges
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    setEdgeList(const Edge<MetaData>* a_edgeList) noexcept;

    /*!
      @brief Set the face list.
      @param[in] a_faceList List (malloc'ed array) of faces
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    setFaceList(const Face<MetaData>* a_faceList) noexcept;

    /*!
      @brief Get the vertex list
      @return m_vertexList
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    const Vertex<MetaData>*
    getVertexList() const noexcept;

    /*!
      @brief Get the edge list
      @return m_edgeList
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    const Edge<MetaData>*
    getEdgeList() const noexcept;

    /*!
      @brief Get the face list.
      @return m_faceList
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    const Face<MetaData>*
    getFaceList() const noexcept;

    /*!
      @brief Normalize the normal vector to a length of 1.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    normalizeNormalVector() noexcept;

    /*!
      @brief Compute the vertex normal, using an average of the normal vectors of all faces
      sharing this vertex. 
      @details This computes the vertex normal as n = sum(normal(face))/num(faces)
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    computeVertexNormalAverage() noexcept;

    /*!
      @brief Compute the vertex normal, using the pseudonormal algorithm which
      weights the normal with the subtended angle to each connected face.
      @details This computes the normal vector using the pseudnormal algorithm from
      Baerentzen and Aanes in "Signed distance computation using the angle
      weighted pseudonormal" (DOI: 10.1109/TVCG.2005.49)
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    computeVertexNormalAngleWeighted() noexcept;

    /*!
      @brief Get the outgoing edge
      @return m_edge
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    int
    getEdge() const noexcept;

    /*!
      @brief Return modifiable vertex position.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Vec3&
    getPosition() noexcept;

    /*!
      @brief Return immutable vertex position.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    const Vec3&
    getPosition() const noexcept;

    /*!
      @brief Return modifiable vertex normal vector.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Vec3&
    getNormal() noexcept;

    /*!
      @brief Return immutable vertex normal vector.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    const Vec3&
    getNormal() const noexcept;

    /*!
      @brief Return outgoing edge index
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    int
    getOutgoingEdge() const noexcept;

    /*!
      @brief Get the signed distance to this vertex
      @param[in] a_x0 Position in space.
      @return The returned distance is |a_x0 - m_position| and the sign is given
      by the sign of m_normal * |a_x0 - m_position|.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Real
    signedDistance(const Vec3& a_x0) const noexcept;

    /*!
      @brief Get the squared unsigned distance to this vertex
      @details This is faster to compute than signedDistance, and might be
      preferred for some algorithms.
      @return Returns the vector length of (a_x - m_position)
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Real
    unsignedDistance2(const Vec3& a_x0) const noexcept;

    /*!
      @brief Get meta-data
      @return m_metaData
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    MetaData&
    getMetaData() noexcept;

    /*!
      @brief Get immutable meta-data
      @return m_metaData
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    const MetaData&
    getMetaData() const noexcept;

  protected:
    /*!
      @brief List of vertices.
    */
    const Vertex<MetaData>* m_vertexList;

    /*!
      @brief List of half-edges.
    */
    const Edge<MetaData>* m_edgeList;

    /*!
      @brief List of faces
    */
    const Face<MetaData>* m_faceList;

    /*!
      @brief Outgoing edge.
    */
    int m_outgoingEdge;

    /*!
      @brief Vertex position
    */
    Vec3 m_position;

    /*!
      @brief Vertex normal vector
    */
    Vec3 m_normal;

    /*!
      @brief MetaData-data for this vertex
    */
    MetaData m_metaData;
  };
} // namespace EBGeometry::DCEL

#include "EBGeometry_DCEL_VertexImplem.hpp" // NOLINT

#endif
