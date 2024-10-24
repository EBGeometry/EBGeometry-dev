/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_DCEL_Face.hpp
  @brief  Declaration of a polygon face class for use in DCEL descriptions of
  polygon tessellations.
  @author Robert Marskar
*/

#ifndef EBGeometry_DCEL_Face
#define EBGeometry_DCEL_Face

// Our includes
#include "EBGeometry_DCEL.hpp"
#include "EBGeometry_DCEL_Polygon2D.hpp"
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_GPUTypes.hpp"
#include "EBGeometry_Macros.hpp"
#include "EBGeometry_Vec.hpp"

namespace EBGeometry {
  namespace DCEL {

    /*!
      @brief Class which represents a polygon face in a double-edge connected list
      (DCEL).
      @details This class is a polygon face in a DCEL mesh. It contains pointer
      storage to one of the half-edges that circulate the inside of the polygon
      face, as well as having a normal vector, a centroid, and an area. This class
      supports signed distance computations. These computations require algorithms
      that compute e.g. the winding number of the polygon, or the number of times a
      ray cast passes through it. Thus, one of its central features is that it can
      be embedded in 2D by projecting it along the cardinal direction of its normal
      vector. To be fully consistent with a DCEL structure the class stores a
      reference to one of its half edges, but for performance reasons it also stores
      references to the other half edges.
      @note To compute the distance from a point to the face one must determine if
      the point projects "inside" or "outside" the polygon. There are several
      algorithms for this, and by default this class uses a crossing number
      algorithm.
    */
    template <class Meta>
    class Face
    {
    public:
      /*!
	@brief Default constructor. Sets the half-edge to zero and the
	inside/outside algorithm to crossing number algorithm
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE
      Face() noexcept;

      /*!
	@brief Partial constructor. Calls default constructor but associates a
	half-edge. Does not leave the face in a fully defined state. 
	@param[in] a_edge Half-edge
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE
      Face(const int a_edge) noexcept;

      /*!
	@brief Partial constructor.
	@details Calls default constructor but sets the normal vector and half-edge
	equal to the other face's (rest is undefined)
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE
      Face(const Face& a_otherFace) noexcept;

      /*!
	@brief Destructor (does nothing)
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE ~Face() noexcept;

      /*!
	@brief Define function which sets the normal vector and half-edge
	@param[in] a_normal Normal vector
	@param[in] a_edge   Half edge
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE void
      define(const Vec3& a_normal, const int a_edge) noexcept;

      /*!
	@brief Reconcile face. This will compute the normal vector, area, centroid,
	and the 2D embedding of the polygon
	@note "Everything" must be set before doing this, i.e. the face must be
	complete with half edges and there can be no dangling edges.
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE void
      reconcile() noexcept;

      /*!
	@brief Set the half edge
	@param[in] a_halfEdge Half edge
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE void
      setHalfEdge(const int a_halfEdge) noexcept;

      /*!
	@brief Set the vertex list.
	@param[in] a_vertexList List (malloc'ed array) of vertices
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE void
      setVertexList(const Vertex<Meta>* const a_vertexList) noexcept;

      /*!
	@brief Set the edge list.
	@param[in] a_edgeList List (malloc'ed array) of edges
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE void
      setEdgeList(const Edge<Meta>* const a_edgeList) noexcept;

      /*!
	@brief Set the face list.
	@param[in] a_faceList List (malloc'ed array) of faces
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE void
      setFaceList(const Face<Meta>* const a_faceList) noexcept;

      /*!
	@brief Set the inside/outside algorithm when determining if a point projects
	to the inside or outside of the polygon.
	@param[in] a_algorithm Desired algorithm
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE void
      setInsideOutsideAlgorithm(const Polygon2D::InsideOutsideAlgorithm& a_algorithm) noexcept;

      /*!
	@brief Get the number of edges in the polygon
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE int
      getNumEdges() const noexcept;

      /*!
	@brief Get the half edge
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE int
      getEdge() const noexcept;

      /*!
	@brief Get modifiable centroid
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE Vec3&
      getCentroid() noexcept;

      /*!
	@brief Get immutable centroid
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE const Vec3&
      getCentroid() const noexcept;

      /*!
	@brief Get modifiable centroid position in specified coordinate direction
	@param[in] a_dir Coordinate direction
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE Real&
      getCentroid(const int a_dir) noexcept;

      /*!
	@brief Get immutable centroid position in specified coordinate direction
	@param[in] a_dir Coordinate direction
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE const Real&
      getCentroid(const int a_dir) const noexcept;

      /*!
	@brief Get modifiable normal vector
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE Vec3&
      getNormal() noexcept;

      /*!
	@brief Get immutable normal vector
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE const Vec3&
      getNormal() const noexcept;

      /*!
	@brief Get meta-data
	@return m_metaData
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE Meta&
      getMetaData() noexcept;

      /*!
	@brief Get meta-data
	@return m_metaData
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE const Meta&
      getMetaData() const noexcept;

      /*!
	@brief Compute the signed distance to a point.
	@param[in] a_x0 Point in space
	@details This algorithm operates by checking if the input point projects to
	the inside of the polygon. If it does then the distance is just the
	projected distance onto the polygon plane and the sign is well-defined.
	Otherwise, we check the distance to the edges of the polygon.
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE Real
      signedDistance(const Vec3& a_x0) const noexcept;

      /*!
	@brief Compute the unsigned squared distance to a point.
	@param[in] a_x0 Point in space
	@details This algorithm operates by checking if the input point projects to
	the inside of the polygon. If it does then the distance is just the
	projected distance onto the polygon plane. Otherwise, we check the distance
	to the edges of the polygon.
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE Real
      unsignedDistance2(const Vec3& a_x0) const noexcept;

      /*!
	@brief Get the lower-left-most coordinate of this polygon face
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE Vec3
      getSmallestCoordinate() const noexcept;

      /*!
	@brief Get the upper-right-most coordinate of this polygon face
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE Vec3
      getHighestCoordinate() const noexcept;

    protected:
      /*!
	@brief List of vertices.
      */
      const Vertex<Meta>* m_vertexList;

      /*!
	@brief List of half-edges.
      */
      const Edge<Meta>* m_edgeList;

      /*!
	@brief List of faces
      */
      const Face<Meta>* m_faceList;

      /*!
	@brief This polygon's half-edge. 
      */
      int m_edge;

      /*!
	@brief Polygon face normal vector
      */
      Vec3 m_normal;

      /*!
	@brief Polygon face centroid position
      */
      Vec3 m_centroid;

      /*!
	@brief Meta-data attached to this face
      */
      Meta m_metaData;

      /*!
	@brief 2D embedding of this polygon. This is the 2D view of the current
	object projected along its normal vector cardinal.
      */
      Polygon2D m_polygon2D;

      /*!
	@brief Algorithm for inside/outside tests
      */
      Polygon2D::InsideOutsideAlgorithm m_poly2Algorithm;

      /*!
	@brief Compute the centroid position of this polygon
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE void
      computeCentroid() noexcept;

      /*!
	@brief Compute the normal position of this polygon
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE void
      computeNormal() noexcept;

      /*!
	@brief Compute the 2D embedding of this polygon
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE void
      computePolygon2D() noexcept;

      /*!
	@brief Normalize the normal vector, ensuring it has a length of 1
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      EBGEOMETRY_ALWAYS_INLINE void
      normalizeNormalVector() noexcept;

      /*!
	@brief Compute the area of this polygon
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE Real
      computeArea() noexcept;

      /*!
	@brief Return the coordinates of all the vertices on this polygon.
	@details This builds a list of all the vertex coordinates and returns it. It is up to the user
	to delete the list of vertex coordinates later (must be done using 'delete[]').
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE Vec3*
      getAllVertexCoordinates() const noexcept;

      /*!
	@brief Compute the projection of a point onto the polygon face plane
	@param[in] a_p Point in space
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE Vec3
      projectPointIntoFacePlane(const Vec3& a_p) const noexcept;

      /*!
	@brief Check if a point projects to inside or outside the polygon face
	@param[in] a_p Point in space
	@return Returns true if a_p projects to inside the polygon and false
	otherwise.
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE bool
      isPointInsideFace(const Vec3& a_p) const noexcept;
    };
  } // namespace DCEL
} // namespace EBGeometry

#include "EBGeometry_DCEL_FaceImplem.hpp"

#endif
