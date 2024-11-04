/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_DCEL_Edge.hpp
  @brief  Declaration of an edge class for use in DCEL descriptions of polygon
  tessellations.
  @author Robert Marskar
*/

#ifndef EBGeometry_DCEL_Edge
#define EBGeometry_DCEL_Edge

// Our includes
#include "EBGeometry_DCEL.hpp"
#include "EBGeometry_DCEL_Vertex.hpp"
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_GPUTypes.hpp"
#include "EBGeometry_Macros.hpp"
#include "EBGeometry_Vec.hpp"

namespace EBGeometry {
  namespace DCEL {

    /*!
      @brief Class which represents a half-edge in a doubly connected edge list. 
      @details This class is used in DCEL functionality which stores polygonal
      surfaces in a mesh. The information contain in an Edge object contains the
      necessary object for logically circulating the inside of a polygon face. This
      means that a polygon face has a double-connected list of half-edges which
      circulate the interior of the face. The Edge object is such a half-edge; it
      represents the outgoing half-edge from a vertex, located such that it can be
      logically represented as a half edge on the "inside" of a polygon face. 
      corresponding half-edge on the other face that shares this edge. Since this
      class is used with DCEL functionality and signed distance fields, this class
      also has a signed distance function and thus a "normal vector".
      
      @note The normal vector is outgoing, i.e. a point x is "outside" if the dot
      product between n and (x - x0) is positive.

      @note This class is GPU-copyable, with the exception of the vertex list which must
      be set to point to the correct place on the GPU.
    */
    template <class MetaData>
    class Edge
    {
    public:
      /*!
	@brief Default constructors. Creates an invalid edge
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE
      Edge() noexcept;

      /*!
	@brief Copy constructor. Copies all vertices, edges, and normal vectors.
	@param[in] a_otherEdge Other edge
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE
      Edge(const Edge& a_otherEdge) noexcept;

      /*!
	@brief Partial constructor. Sets the starting vertex
	@param[in] a_vertex Starting vertex index.
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE
      Edge(const int a_vertex) noexcept;

      /*!
	@brief Partial constructor. Sets everything except the normal vector and internal lists.
	@param[in] a_vertex Starting vertex index
	@param[in] a_previousEdge Previous edge index.
	@param[in] a_pairEdge Pair edge index.
	@param[in] a_nextEdge Next edge index.
	@param[in] a_face Polygon face index.
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE
      Edge(const int a_vertex,
           const int a_previousEdge,
           const int a_pairEdge,
           const int a_nextEdge,
           const int a_face) noexcept;

      /*!
	@brief Destructor (does nothing).
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE
      ~Edge() noexcept;

      /*!
	@brief Define function. Sets everything except the normal vector.
	@param[in] a_vertex Starting vertex index
	@param[in] a_previousEdge Previous edge index.
	@param[in] a_pairEdge Pair edge index.
	@param[in] a_nextEdge Next edge index.
	@param[in] a_face Polygon face index.
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE
      void
      define(const int a_vertex,
             const int a_previousEdge,
             const int a_pairEdge,
             const int a_nextEdge,
             const int a_face) noexcept;

      /*!
	@brief Set the starting vertex
	@param[in] a_vertex Starting vertex.
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE
      void
      setVertex(const int a_vertex) noexcept;

      /*!
	@brief Set the previous edge
	@param[in] a_previousEdge Previous edge
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE
      void
      setPreviousEdge(const int a_previousEdge) noexcept;

      /*!
	@brief Set the pair edge
	@param[in] a_pairEdge Pair edge (for jumping to the opposite polygon)
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE
      void
      setPairEdge(const int a_pairEdge) noexcept;

      /*!
	@brief Set the next edge
	@param[in] a_nextEdge Next edge around the polygon
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE
      void
      setNextEdge(const int a_nextEdge) noexcept;

      /*!
	@brief Set the polygon face.
	@param[in] a_face Polygon face.
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE
      void
      setFace(const int a_face) noexcept;

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
      setVertexList(const Vertex<MetaData>* const a_vertexList) noexcept;

      /*!
	@brief Set the edge list.
	@param[in] a_edgeList List (malloc'ed array) of edges
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE
      void
      setEdgeList(const Edge<MetaData>* const a_edgeList) noexcept;

      /*!
	@brief Set the face list.
	@param[in] a_faceList List (malloc'ed array) of faces
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE
      void
      setFaceList(const Face<MetaData>* const a_faceList) noexcept;

      /*!
	@brief Get the vertex list
	@return m_vertexList
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE
      const Vertex<MetaData>*
      getVertexList() const noexcept;

      /*!
	@brief Get the edge list
	@return m_edgeList
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE
      const Edge<MetaData>*
      getEdgeList() const noexcept;

      /*!
	@brief Get the face list.
	@return m_faceList
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE
      const Face<MetaData>*
      getFaceList() const noexcept;

      /*!
	@brief Set the normal vector
	@param[in] a_normal Normal vector
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE
      void
      setNormal(const Vec3& a_normal) noexcept;

      /*!
	@brief Normalize the normal vector, ensuring it has a length of 1
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE
      void
      normalizeNormalVector() noexcept;

      /*!
	@brief Compute the normal vector
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE
      void
      computeNormal() noexcept;

      /*!
	@brief Get starting vertex
	@return Returns m_vertex
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
      int
      getVertex() const noexcept;

      /*!
	@brief Get the end vertex
	@return Returns m_vertex
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
      int
      getOtherVertex() const noexcept;

      /*!
	@brief Get previous edge
	@return Returns m_previousEdge
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
      int
      getPreviousEdge() const noexcept;

      /*!
	@brief Get pair edge
	@return Returns m_pairEdge
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
      int
      getPairEdge() const noexcept;

      /*!
	@brief Get next edge
	@return Returns m_nextEdge
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
      int
      getNextEdge() const noexcept;

      /*!
	@brief Get polygon face.
	@return Returns m_face
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
      int
      getFace() const noexcept;

      /*!
	@brief Get the normal vector
	@return Return m_normal
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
      const Vec3&
      getNormal() const noexcept;

      /*!
	@brief Get meta-data
	@return m_metaData
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
      MetaData&
      getMetaData() noexcept;

      /*!
	@brief Get meta-data
	@return m_metaData
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
      const MetaData&
      getMetaData() const noexcept;

      /*!
	@brief Get the signed distance to this half edge
	@details This routine will check if the input point projects to the edge or
	one of the vertices. If it projectes to one of the vertices we compute the
	signed distance to the corresponding vertex. Otherwise we compute the
	projection to the edge and compute the sign from the normal vector.
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
      Real
      signedDistance(const Vec3& a_x0) const noexcept;

      /*!
	@brief Get the signed distance to this half edge
	@details This routine will check if the input point projects to the edge or
	one of the vertices. If it projectes to one of the vertices we compute the
	squared distance to the corresponding vertex. Otherwise we compute the
	squared distance of the projection to the edge. This is faster than
	signedDistance()
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
      Real
      unsignedDistance2(const Vec3& a_x0) const noexcept;

    protected:
      /*!
	@brief Vertex list
      */
      const Vertex<MetaData>* m_vertexList;

      /*!
	@brief Edge list
      */
      const Edge<MetaData>* m_edgeList;

      /*!
	@brief Face list
      */
      const Face<MetaData>* m_faceList;

      /*!
	@brief Starting vertex.
      */
      int m_vertex;

      /*!
	@brief Previous edge
      */
      int m_previousEdge;

      /*!
	@brief Pair edge.
      */
      int m_pairEdge;

      /*!
	@brief Next edge around the polygon.
      */
      int m_nextEdge;

      /*!
	@brief Polygon face connected to this half-edge.
      */
      int m_face;

      /*!
	@brief Edge normal vector
      */
      Vec3 m_normal;

      /*!
	@brief MetaData-data attached to this edge
      */
      MetaData m_metaData;

      /*!
	@brief Return the vector pointing along this edge.
	@details Returns the vector pointing from the starting index to the end index
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE
      Vec3
      getX2X1() const noexcept;

      /*!
	@brief Returns the "projection" of a point to an edge.
	@details This function parametrizes the edge as x(t) = x0 + (x1-x0)*t and
	returns where on the this edge the point a_x0 projects. If projects onto the
	edge if t = [0,1] and to one of the start/end vertices otherwise.
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE
      Real
      projectPointToEdge(const Vec3& a_x0) const noexcept;
    };

  } // namespace DCEL
} // namespace EBGeometry

#include "EBGeometry_DCEL_EdgeImplem.hpp"

#endif
