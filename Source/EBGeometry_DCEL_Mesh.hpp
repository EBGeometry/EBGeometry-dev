/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_DCEL_Mesh.hpp
  @brief  Declaration of a mesh class which stores a DCEL mesh (with signed
  distance functions)
  @author Robert Marskar
*/

#ifndef EBGeometry_DCEL_Mesh
#define EBGeometry_DCEL_Mesh

// Std includes
#include <map>

// Our includes
#include "EBGeometry_DCEL.hpp"
#include "EBGeometry_DCEL_Edge.hpp"
#include "EBGeometry_DCEL_Face.hpp"
#include "EBGeometry_DCEL_Vertex.hpp"
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_GPUTypes.hpp"

namespace EBGeometry {
  namespace DCEL {

    /*!
      @brief Mesh class which stores a full DCEL mesh (with signed distance
      functions)
      @details This encapsulates a full DCEL mesh, and also includes direct signed
      distance functions. The mesh consists of a set of vertices, half-edges, and
      polygon faces who each have references to other vertices, half-edges, and
      polygon faces. The signed distance functions are direct, which means that they go
      through all of the polygon faces and compute the signed distance to them. This
      is extremely inefficient, which is why these meshes are almost always embedded
      into a bounding volume hierarchy.

      Note that the Mesh class does not own the DCEL mesh data, and is only given
      pointer access to the raw data.
      
      @note This class is not for the light of heart -- it will almost always be
      instantiated through a file parser which reads vertices and edges from file
      and builds the mesh from that. Do not try to build a Mesh object yourself,
      use file parsers!
    */
    template <class Meta>
    class Mesh
    {
    public:
      /*!
	@brief Possible search algorithms for DCEL::Mesh
	@details Direct means compute the signed distance for all primitives,
	Direct2 means compute the squared signed distance for all primitives.
      */
      enum class SearchAlgorithm
      {
        Direct,
        Direct2,
      };

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
	@brief Default constructor. Leaves unobject in an unusable state
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline Mesh() noexcept;

      /*!
	@brief Disallowed copy construction
	@param[in] a_otherMesh Other mesh
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline Mesh(const Mesh& a_otherMesh) = delete;

      /*!
	@brief Full constructor. This provides the faces, edges, and vertices to the
	mesh.
	@param[in] a_numFaces Number of faces in face list
	@param[in] a_numEdges Number of edges in edge list
	@param[in] a_numVertices Number of vertices in vertex list	
	@param[in] a_faces Polygon faces
	@param[in] a_edges Half-edges
	@param[in] a_vertices Vertices
	@note The constructor arguments should provide a complete DCEL mesh
	description. This is usually done through a file parser which reads a mesh
	file format and creates the DCEL mesh structure.
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline Mesh(const int           a_numFaces,
                  const int           a_numEdges,
                  const int           a_numVertices,
                  const FacePointer   a_faces,
                  const EdgePointer   a_edges,
                  const VertexPointer a_vertices) noexcept;

      /*!
	@brief Destructor (does nothing)
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline ~Mesh() noexcept;

      /*!
	@brief Define function. Puts Mesh in usable state.
	@param[in] a_numFaces Number of faces in face list
	@param[in] a_numEdges Number of edges in edge list
	@param[in] a_numVertices Number of vertices in vertex list	
	@param[in] a_faces Polygon faces
	@param[in] a_edges Half-edges
	@param[in] a_vertices Vertices
	@note The function arguments should provide a complete DCEL mesh
	description. This is usually done through a file parser which reads a mesh
	file format and creates the DCEL mesh structure. Note that this only
	involves associating pointer structures through the mesh. Internal
	parameters for the faces, edges, and vertices are computed through the
	reconcile function (which is called by Mesh<Meta>::define).
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline void
      define(const int           a_numFaces,
             const int           a_numEdges,
             const int           a_numVertices,
             const FacePointer   a_faces,
             const EdgePointer   a_edges,
             const VertexPointer a_vertices) noexcept;

      /*!
	@brief Perform a sanity check.
	@details This will provide error messages if vertices are badly linked,
	faces are nullptr, and so on. These messages are logged by calling
	incrementWarning() which identifies types of errors that can occur, and how
	many of those errors have occurred.
      */
      inline void
      sanityCheck() const noexcept;

      /*!
	@brief Search algorithm for direct signed distance computations
	@param[in] a_algorithm Algorithm to use
      */
      inline void
      setSearchAlgorithm(const SearchAlgorithm a_algorithm) noexcept;

      /*!
	@brief Set the inside/outside algorithm to use when computing the signed
	distance to polygon faces.
	@details Computing the signed distance to faces requires testing if a point
	projected to a polygo face plane falls inside or outside the polygon face.
	There are multiple algorithms to use here.
	@param[in] a_algorithm Algorithm to use
      */
      inline void
      setInsideOutsideAlgorithm(typename Polygon2D<T>::InsideOutsideAlgorithm a_algorithm) noexcept;

      /*!
	@brief Reconcile function which computes the internal parameters in
	vertices, edges, and faces for use with signed distance functionality
	@param[in] a_weight Vertex angle weighting function. Either
	VertexNormalWeight::None for unweighted vertex normals or
	VertexNormalWeight::Angle for the pseudonormal
	@details This will reconcile faces, edges, and vertices, e.g. computing the
	area and normal vector for faces
      */
      inline void
      reconcile(const DCEL::VertexNormalWeight a_weight = DCEL::VertexNormalWeight::Angle) noexcept;

      /*!
	@brief Flip the mesh, making all the normals change direction. 
	@note Should be called AFTER all normals have been computed. 
      */
      inline void
      flip() noexcept;

      /*!
	@brief Get modifiable vertices in this mesh
      */
      inline std::vector<VertexPtr>&
      getVertices() noexcept;

      /*!
	@brief Get immutable vertices in this mesh
      */
      inline const std::vector<VertexPtr>&
      getVertices() const noexcept;

      /*!
	@brief Return all vertex coordinates in the mesh.
      */
      inline std::vector<Vec3T<T>>
      getAllVertexCoordinates() const noexcept;

      /*!
	@brief Get modifiable half-edges in this mesh
      */
      inline std::vector<EdgePtr>&
      getEdges() noexcept;

      /*!
	@brief Get immutable half-edges in this mesh
      */
      inline const std::vector<EdgePtr>&
      getEdges() const noexcept;

      /*!
	@brief Get modifiable faces in this mesh
      */
      inline std::vector<FacePtr>&
      getFaces() noexcept;

      /*!
	@brief Get immutable faces in this mesh
      */
      inline const std::vector<FacePtr>&
      getFaces() const noexcept;

      /*!
	@brief Compute the signed distance from a point to this mesh
	@param[in] a_x0 3D point in space.
	@details This function will iterate through ALL faces in the mesh and return
	the value with the smallest magnitude. This is horrendously slow, which is
	why this function is almost never called. Rather, Mesh<T, Meta> can be embedded
	in a bounding volume hierarchy for faster access.
	@note This will call the other version with the object's search algorithm.
      */
      inline T
      signedDistance(const Vec3& a_x0) const noexcept;

      /*!
	@brief Compute the signed distance from a point to this mesh
	@param[in] a_x0        3D point in space.
	@param[in] a_algorithm Search algorithm
	@details This function will iterate through ALL faces in the mesh and return
	the value with the smallest magnitude. This is horrendously slow, which is
	why this function is almost never called. Rather, Mesh<T, Meta> can be embedded
	in a bounding volume hierarchy for faster access.
      */
      inline T
      signedDistance(const Vec3& a_x0, SearchAlgorithm a_algorithm) const noexcept;

      /*!
	@brief Compute the unsigned square distance from a point to this mesh
	@param[in] a_x0 3D point in space.
	@details This function will iterate through ALL faces in the mesh and return
	the value with the smallest magnitude. This is horrendously slow, which is
	why this function is almost never called. Rather, Mesh<T, Meta> can be embedded
	in a bounding volume hierarchy for faster access.
	@note This will call the other version with the object's search algorithm.
      */
      inline T
      unsignedDistance2(const Vec3& a_x0) const noexcept;

    protected:
      /*!
	@brief Search algorithm. Only used in signed distance functions.
      */
      SearchAlgorithm m_algorithm;

      /*!
	@brief Mesh vertices
      */
      std::vector<VertexPtr> m_vertices;

      /*!
	@brief Mesh half-edges
      */
      std::vector<EdgePtr> m_edges;

      /*!
	@brief Mesh faces
      */
      std::vector<FacePtr> m_faces;

      /*!
	@brief Function which computes internal things for the polygon faces.
	@note This calls DCEL::FaceT<T, Meta>::reconcile()
      */
      inline void
      reconcileFaces() noexcept;

      /*!
	@brief Function which computes internal things for the half-edges
	@note This calls DCEL::EdgeT<T, Meta>::reconcile()
      */
      inline void
      reconcileEdges() noexcept;

      /*!
	@brief Function which computes internal things for the vertices
	@param[in] a_weight Vertex angle weighting
	@note This calls DCEL::VertexT<T, Meta>::computeVertexNormalAverage() or
	DCEL::VertexT<T, Meta>::computeVertexNormalAngleWeighted()
      */
      inline void
      reconcileVertices(const DCEL::VertexNormalWeight a_weight) noexcept;

      /*!
	@brief Flip all face normals
      */
      inline void
      flipFaces() noexcept;

      /*!
	@brief Flip all edge normals
      */
      inline void
      flipEdges() noexcept;

      /*!
	@brief Flip all vertex normal
      */
      inline void
      flipVertices() noexcept;

      /*!
	@brief Implementation of signed distance function which iterates through all
	faces
	@param[in] a_point 3D point
      */
      inline T
      DirectSignedDistance(const Vec3& a_point) const noexcept;

      /*!
	@brief Implementation of squared signed distance function which iterates
	through all faces.
	@details This first find the face with the smallest unsigned square
	distance, and the returns the signed distance to that face (more efficient
	than the other version).
	@param[in] a_point 3D point
      */
      inline T
      DirectSignedDistance2(const Vec3& a_point) const noexcept;

      /*!
	@brief Increment a warning. This is used in sanityCheck() for locating holes
	or bad inputs in the mesh.
	@param[in] a_warnings Map of all registered warnings
	@param[in] a_warn     Current warning to increment by
      */
      inline void
      incrementWarning(std::map<std::string, size_t>& a_warnings, const std::string& a_warn) const noexcept;

      /*!
	@brief Print all warnings to std::cerr
      */
      inline void
      printWarnings(const std::map<std::string, size_t>& a_warnings) const noexcept;
    };
  } // namespace DCEL
}

//#include "EBGeometry_DCEL_MeshImplem.hpp"

#endif
