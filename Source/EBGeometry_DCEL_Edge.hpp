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
      @brief Class which represents a half-edge in a double-edge connected list
      (DCEL).
      @details This class is used in DCEL functionality which stores polygonal
      surfaces in a mesh. The information contain in an Edge object contains the
      necessary object for logically circulating the inside of a polygon face. This
      means that a polygon face has a double-connected list of half-edges which
      circulate the interior of the face. The Edge object is such a half-edge; it
      represents the outgoing half-edge from a vertex, located such that it can be
      logically represented as a half edge on the "inside" of a polygon face. It
      contains pointers to the polygon face, vertex, and next half edge It also contains 
      a pointer to the "pair" half edge, i.e. the
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

    protected:
      /*!
	@brief Edge normal vector
      */
      Vec3 m_normal;

      /*!
	@brief Index of starting vertex in vertex list
      */
      int m_vertex;

      /*!
	@brief Index of pair edge in pair list. 
      */
      int m_pairEdge;

      /*!
	@brief Index of next edge in edge list. 
      */
      int m_nextEdge;

      /*!
	@brief Index of enclosing polygon face in face list.
      */
      int m_face;

      /*!
	@brief Meta-data attached to this edge
      */
      Meta m_metaData;

      /*!
	@brief Returns the "projection" of a point to an edge.
	@details This function parametrizes the edge as x(t) = x0 + (x1-x0)*t and
	returns where on the this edge the point a_x0 projects. If projects onto the
	edge if t = [0,1] and to one of the start/end vertices otherwise.
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline Real
      projectPointToEdge(const Vec3& a_x0) const noexcept;

      /*!
	@brief Return the vector pointing along this edge.
	@details Returns the vector pointing from the starting index to the end index
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline Vec3
      getX2X1() const noexcept;
    };

  } // namespace DCEL
} // namespace EBGeometry

#include "EBGeometry_DCEL_VertexImplem.hpp"

#endif
