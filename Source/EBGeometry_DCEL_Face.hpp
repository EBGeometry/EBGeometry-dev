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

namespace EBGeometry::DCEL {

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
  template <class MetaData>
  class Face
  {
  public:
    /*!
      @brief Default constructor. Sets the half-edge to zero and the
      inside/outside algorithm to crossing number algorithm
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Face() noexcept = default;

    /*!
      @brief Partial constructor. Calls default constructor but associates a
      half-edge. Does not leave the face in a fully defined state. 
      @param[in] a_edge Half-edge
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Face(int a_edge) noexcept;

    /*!
      @brief Copy constructor.
      @param[in] a_otherFace Other face
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Face(const Face& a_otherFace) noexcept = default;

    /*!
      @brief Move constructor.
      @param[in, out] a_otherFace Other face
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Face(Face&& a_otherFace) noexcept = default;

    /*!
      @brief Destructor (does nothing)
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    ~Face() noexcept = default;

    /*!
      @brief Copy assignment
      @param[in] a_face Other face
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Face&
    operator=(const Face& a_face) noexcept = default;

    /*!
      @brief Move assignment
      @param[in] a_face Other face
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Face&
    operator=(Face&& a_face) noexcept = default;

    /*!
      @brief Reconcile face. This will compute the normal vector, area, centroid,
      and the 2D embedding of the polygon
      @note "Everything" must be set before doing this, i.e. the face must be
      complete with half edges and there can be no dangling edges.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    reconcile() noexcept;

    /*!
      @brief Set the half edge
      @param[in] a_edge Half edge
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
      @brief Set the inside/outside algorithm when determining if a point projects
      to the inside or outside of the polygon.
      @param[in] a_algorithm Desired algorithm
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    setInsideOutsideAlgorithm(const Polygon2D::InsideOutsideAlgorithm& a_algorithm) noexcept;

    /*!
      @brief Get the number of edges in the polygon
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    int
    getNumEdges() const noexcept;

    /*!
      @brief Get the half edge
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    int
    getEdge() const noexcept;

    /*!
      @brief Get modifiable centroid
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Vec3&
    getCentroid() noexcept;

    /*!
      @brief Get immutable centroid
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    const Vec3&
    getCentroid() const noexcept;

    /*!
      @brief Get modifiable centroid position in specified coordinate direction
      @param[in] a_dir Coordinate direction
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Real&
    getCentroid(int a_dir) noexcept;

    /*!
      @brief Get immutable centroid position in specified coordinate direction
      @param[in] a_dir Coordinate direction
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    const Real&
    getCentroid(int a_dir) const noexcept;

    /*!
      @brief Get modifiable normal vector
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Vec3&
    getNormal() noexcept;

    /*!
      @brief Get immutable normal vector
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
      @brief Compute the signed distance to a point.
      @param[in] a_x0 Point in space
      @details This algorithm operates by checking if the input point projects to
      the inside of the polygon. If it does then the distance is just the
      projected distance onto the polygon plane and the sign is well-defined.
      Otherwise, we check the distance to the edges of the polygon.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Real
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
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Real
    unsignedDistance2(const Vec3& a_x0) const noexcept;

    /*!
      @brief Get the lower-left-most coordinate of this polygon face
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Vec3
    getSmallestCoordinate() const noexcept;

    /*!
      @brief Get the upper-right-most coordinate of this polygon face
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Vec3
    getHighestCoordinate() const noexcept;

  protected:
    /*!
      @brief List of vertices.
    */
    const Vertex<MetaData>* m_vertexList = nullptr;

    /*!
      @brief List of half-edges.
    */
    const Edge<MetaData>* m_edgeList = nullptr;

    /*!
      @brief List of faces
    */
    const Face<MetaData>* m_faceList = nullptr;

    /*!
      @brief This polygon's half-edge. 
    */
    int m_edge = -1;

    /*!
      @brief Polygon face normal vector
    */
    Vec3 m_normal = Vec3::zero();

    /*!
      @brief Polygon face centroid position
    */
    Vec3 m_centroid = Vec3::zero();

    /*!
      @brief MetaData-data attached to this face
    */
    MetaData m_metaData = MetaData();

    /*!
      @brief 2D embedding of this polygon. This is the 2D view of the current
      object projected along its normal vector cardinal.
    */
    Polygon2D m_polygon2D;

    /*!
      @brief Algorithm for inside/outside tests
    */
    Polygon2D::InsideOutsideAlgorithm m_poly2Algorithm = Polygon2D::InsideOutsideAlgorithm::CrossingNumber;

    /*!
      @brief Compute the centroid position of this polygon
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    computeCentroid() noexcept;

    /*!
      @brief Compute the normal position of this polygon
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    computeNormal() noexcept;

    /*!
      @brief Compute the 2D embedding of this polygon
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    computePolygon2D() noexcept;

    /*!
      @brief Normalize the normal vector, ensuring it has a length of 1
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    normalizeNormalVector() noexcept;

    /*!
      @brief Compute the area of this polygon
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Real
    computeArea() noexcept;

    /*!
      @brief Return the coordinates of all the vertices on this polygon.
      @details This builds a list of all the vertex coordinates and returns it. It is up to the user
      to delete the list of vertex coordinates later (must be done using 'delete[]').
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Vec3*
    getAllVertexCoordinates() const noexcept;

    /*!
      @brief Compute the projection of a point onto the polygon face plane
      @param[in] a_p Point in space
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Vec3
    projectPointIntoFacePlane(const Vec3& a_p) const noexcept;

    /*!
      @brief Check if a point projects to inside or outside the polygon face
      @param[in] a_p Point in space
      @return Returns true if a_p projects to inside the polygon and false
      otherwise.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    bool
    isPointInsideFace(const Vec3& a_p) const noexcept;
  };
} // namespace EBGeometry::DCEL

#include "EBGeometry_DCEL_FaceImplem.hpp" // NOLINT

#endif
