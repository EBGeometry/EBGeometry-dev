/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_DCEL_Edge.hpp
  @brief  Declaration of an edge class for use in DCEL descriptions of polygon
  tesselations.
  @author Robert Marskar
*/

#ifndef EBGeometry_DCEL_Edge
#define EBGeometry_DCEL_Edge

// Our includes
#include "EBGeometry_DCEL.hpp"
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_GPUTypes.hpp"
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
    */
    template <class Meta>
    class Edge
    {
    public:
      /*!
	@brief Alias for a vertex
      */
      using VertexPointer = const Vertex<Meta>*;

      /*!
	@brief Alias for a half-edge
      */
      using EdgePointer = const Edge<Meta>*;

      /*!
	@brief Pointer to a polygon face
      */
      using FacePointer = const Face<Meta>*;

      /*!
	@brief Default constructors. Creates an invalid edge
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline Edge() noexcept;

      /*!
	@brief Copy constructor. Copies all vertices, edges, and normal vectors.
	@param[in] a_otherEdge Other edge
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline Edge(const Edge& a_otherEdge) noexcept;

      /*!
	@brief Partial constructor. Sets the starting vertex
	@param[in] a_vertex Starting vertex.
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline Edge(const VertexPointer a_vertex) noexcept;

      /*!
	@brief Partial constructor. Sets everything except the normal vector.
	@param[in] a_vertex Starting vertex
	@param[in] a_pairEdge Pair edge (for jumping to opposite polygon)
	@param[in] a_nextEdge Next edge around the polygon.
	@param[in] a_face Polygon face
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline Edge(const VertexPointer a_vertex,
                  const EdgePointer   a_pairEdge,
                  const EdgePointer   a_nextEdge,
                  const FacePointer   a_face) noexcept;

      /*!
	@brief Destructor (does nothing).
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline ~Edge() noexcept;

      /*!
	@brief Define function. Sets everything except the normal vector.
	@param[in] a_vertex Starting vertex
	@param[in] a_pairEdge Pair edge (for jumping to opposite polygon)
	@param[in] a_nextEdge Next edge around the polygon.
	@param[in] a_face Polygon face
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline void
      define(const VertexPointer a_vertex,
             const EdgePointer   a_pairEdge,
             const EdgePointer   a_nextEdge,
             const FacePointer   a_face) noexcept;

      /*!
	@brief Set the starting vertex
	@param[in] a_vertex Starting vertex.
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline void
      setVertex(const VertexPointer a_vertex) noexcept;

      /*!
	@brief Set the pair edge
	@param[in] a_pairEdge Pair edge (for jumping to the opposite polygon)
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline void
      setPairEdge(const EdgePointer a_pairEdge) noexcept;

      /*!
	@brief Set the next edge
	@param[in] a_nextEdge Next edge around the polygon
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline void
      setNextEdge(const EdgePointer a_nextEdge) noexcept;

      /*!
	@brief Set the polygon face.
	@param[in] a_face Polygon face.
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline void
      setFace(const FacePointer a_face) noexcept;

      /*!
	@brief Set the normal vector
	@param[in] a_normal Normal vector
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline void
      setNormal(const Vec3& a_normal) noexcept;

      /*!
	@brief Get starting vertex
	@return Returns m_vertex
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] inline VertexPointer
      getVertex() const noexcept;

      /*!
	@brief Get the end vertex
	@return Returns m_vertex
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] inline VertexPointer
      getOtherVertex() const noexcept;

      /*!
	@brief Get pair edge
	@return Returns m_pairEdge
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] inline EdgePointer
      getPairEdge() const noexcept;

      /*!
	@brief Get next edge
	@return Returns m_nextEdge
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] inline EdgePointer
      getNextEdge() const noexcept;

      /*!
	@brief Get polygon face.
	@return Returns m_face
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] inline FacePointer
      getFace() const noexcept;

      /*!
	@brief Get the normal vector
	@return Return m_normal
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] inline const Vec3&
      getNormal() const noexcept;

      /*!
	@brief Get meta-data
	@return m_metaData
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] inline Meta&
      getMetaData() noexcept;

      /*!
	@brief Get meta-data
	@return m_metaData
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] inline const Meta&
      getMetaData() const noexcept;

      /*!
	@brief Get the signed distance to this half edge
	@details This routine will check if the input point projects to the edge or
	one of the vertices. If it projectes to one of the vertices we compute the
	signed distance to the corresponding vertex. Otherwise we compute the
	projection to the edge and compute the sign from the normal vector.
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] inline Real
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
      [[nodiscard]] inline Real
      unsignedDistance2(const Vec3& a_x0) const noexcept;

    protected:
      /*!
	@brief Starting vertex.
      */
      VertexPointer m_vertex;

      /*!
	@brief Pair edge.
      */
      EdgePointer m_pairEdge;

      /*!
	@brief Next edge around the polygon.
      */
      EdgePointer m_nextEdge;

      /*!
	@brief Polygon face connected to this half-edge.
      */
      FacePointer m_face;

      /*!
	@brief Edge normal vector
      */
      Vec3 m_normal;

      /*!
	@brief Meta-data attached to this edge
      */
      Meta m_metaData;

      /*!
	@brief Return the vector pointing along this edge.
	@details Returns the vector pointing from the starting index to the end index
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline Vec3
      getX2X1() const noexcept;

      /*!
	@brief Returns the "projection" of a point to an edge.
	@details This function parametrizes the edge as x(t) = x0 + (x1-x0)*t and
	returns where on the this edge the point a_x0 projects. If projects onto the
	edge if t = [0,1] and to one of the start/end vertices otherwise.
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline Real
      projectPointToEdge(const Vec3& a_x0) const noexcept;
    };

  } // namespace DCEL
} // namespace EBGeometry

#include "EBGeometry_DCEL_EdgeImplem.hpp"

#endif
