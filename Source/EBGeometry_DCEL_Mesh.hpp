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
#include "EBGeometry_DCEL_Polygon2D.hpp"
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
      pointer access to the raw data. However, Mesh IS permitted to modify the DCEL
      structures (vertices, edges, faces).
      
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
	@brief Default constructor. Leaves unobject in an unusable state
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline Mesh() noexcept;

      /*!
	@brief Full constructor. This provides the faces, edges, and vertices to the
	mesh.
	@param[in] a_numVertices Number of vertices in vertex list
	@param[in] a_numEdges Number of edges in edge list	
	@param[in] a_numFaces Number of faces in face list	
	@param[in] a_vertices Vertices
	@param[in] a_edges Half-edges	
	@param[in] a_faces Polygon faces	
	@note The constructor arguments should provide a complete DCEL mesh
	description. This is usually done through a file parser which reads a mesh
	file format and creates the DCEL mesh structure.
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline Mesh(const int     a_numVertices,
                  const int     a_numEdges,
                  const int     a_numFaces,
                  Vertex<Meta>* a_vertices,
                  Edge<Meta>*   a_edges,
                  Face<Meta>*   a_faces) noexcept;

      /*!
	@brief Destructor (does nothing)
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline ~Mesh() noexcept;

      /*!
	@brief Define function. Puts mesh in usable state. 
	@param[in] a_numVertices Number of vertices in vertex list
	@param[in] a_numEdges Number of edges in edge list	
	@param[in] a_numFaces Number of faces in face list	
	@param[in] a_vertices Vertices
	@param[in] a_edges Half-edges
	@param[in] a_faces Polygon faces	
	@note The function arguments should provide a complete DCEL mesh
	description. This is usually done through a file parser which reads a mesh
	file format and creates the DCEL mesh structure. Note that this only
	involves associating pointer structures through the mesh. Internal
	parameters for the faces, edges, and vertices are computed through the
	reconcile function (which is called by Mesh<Meta>::define).
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline void
      define(const int     a_numVertices,
             const int     a_numEdges,
             const int     a_numFaces,
             Vertex<Meta>* a_vertices,
             Edge<Meta>*   a_edges,
             Face<Meta>*   a_faces) noexcept;

      /*!
	@brief Perform a sanity check.
	@details This will provide error messages if vertices are badly linked,
	faces are nullptr, and so on. These messages are logged by calling
	incrementWarning() which identifies types of errors that can occur, and how
	many of those errors have occurred.
	@note Only callable on host. 
      */
      EBGEOMETRY_GPU_HOST
      inline void
      sanityCheck() const noexcept;

      /*!
	@brief Search algorithm for direct signed distance computations
	@param[in] a_algorithm Algorithm to use.
      */
      EBGEOMETRY_GPU_HOST_DEVICE
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
      EBGEOMETRY_GPU_HOST_DEVICE
      inline void
      setInsideOutsideAlgorithm(const Polygon2D::InsideOutsideAlgorithm a_algorithm) noexcept;

      /*!
	@brief Reconcile function which computes the internal parameters in
	vertices, edges, and faces for use with signed distance functionality
	@param[in] a_weight Vertex angle weighting function. Either
	VertexNormalWeight::None for unweighted vertex normals or
	VertexNormalWeight::Angle for the pseudonormal
	@details This will reconcile faces, edges, and vertices, e.g. computing the
	area and normal vector for faces
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline void
      reconcile(const DCEL::VertexNormalWeight a_weight = DCEL::VertexNormalWeight::Angle) noexcept;

      /*!
	@brief Get vertices in this mesh
	@return m_vertices
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] inline const Vertex<Meta>*
      getVertices() const noexcept;

      /*!
	@brief Get number of vertices in this mesh
	@return m_numVertices
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] inline int
      getNumberOfVertices() const noexcept;

      /*!
	@brief Get half-edges in this mesh
	@return m_edges
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] inline const Edge<Meta>*
      getEdges() const noexcept;

      /*!
	@brief Get number of half-edges in thius mesh
	@return m_numEdges
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] inline int
      getNumberOfEdges() const noexcept;

      /*!
	@brief Get faces in this mesh.
	@return m_faces
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] inline const Face<Meta>*
      getFaces() const noexcept;

      /*!
	@brief Get number of faces in this mesh.
	@return m_numFaces
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] inline int
      getNumberOfFaces() const noexcept;

      /*!
	@brief Compute the signed distance from a point to this mesh
	@param[in] a_point 3D point in space.
	@details This function will iterate through ALL faces in the mesh and return
	the value with the smallest magnitude. This is horrendously slow, which is
	why this function is almost never called. Rather, Mesh<Meta> can be embedded
	in a bounding volume hierarchy for faster access.
	@note This will call the other version with the object's search algorithm.
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] inline Real
      signedDistance(const Vec3& a_point) const noexcept;

      /*!
	@brief Compute the signed distance from a point to this mesh
	@details This function will iterate through ALL faces in the mesh and return
	the value with the smallest magnitude. This is horrendously slow, which is
	why this function is almost never called. Rather, Mesh<Meta> can be embedded
	in a bounding volume hierarchy for faster access.	
	@param[in] a_point 3D point in space.
	@param[in] a_algorithm Search algorithm

      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] inline Real
      signedDistance(const Vec3& a_point, SearchAlgorithm a_algorithm) const noexcept;

      /*!
	@brief Compute the unsigned square distance from a point to this mesh
	@param[in] a_point 3D point in space.
	@details This function will iterate through ALL faces in the mesh and return
	the value with the smallest magnitude. This is horrendously slow, which is
	why this function is almost never called. Rather, Mesh<Meta> can be embedded
	in a bounding volume hierarchy for faster access.
	@note This will call the other version with the object's search algorithm.
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] inline Real
      unsignedDistance2(const Vec3& a_point) const noexcept;

    protected:
      /*!
	@brief Search algorithm. Only used in signed distance functions.
      */
      SearchAlgorithm m_algorithm;

      /*!
	@brief Vertex list
      */
      Vertex<Meta>* m_vertices;

      /*!
	@brief Edge list
      */
      Edge<Meta>* m_edges;

      /*!
	@brief Face list
      */
      Face<Meta>* m_faces;

      /*!
	@brief Number of vertices
      */
      int m_numVertices;

      /*!
	@brief Number of edges
      */
      int m_numEdges;

      /*!
	@brief Number of faces
      */
      int m_numFaces;

      /*!
	@brief Function which computes internal things for the polygon faces.
	@note This calls DCEL::Face<Meta>::reconcile()
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline void
      reconcileFaces() noexcept;

      /*!
	@brief Function which computes internal things for the half-edges
	@note This calls DCEL::Edge<Meta>::reconcile()
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline void
      reconcileEdges() noexcept;

      /*!
	@brief Function which computes internal things for the vertices
	@param[in] a_weight Vertex angle weighting
	@note This calls DCEL::Vertex<Meta>::computeVertexNormalAverage() or
	DCEL::Vertex<Meta>::computeVertexNormalAngleWeighted()
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      inline void
      reconcileVertices(const DCEL::VertexNormalWeight a_weight) noexcept;

      /*!
	@brief Implementation of signed distance function which iterates through all
	faces
	@param[in] a_point 3D point
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] inline Real
      DirectSignedDistance(const Vec3& a_point) const noexcept;

      /*!
	@brief Implementation of squared signed distance function which iterates
	through all faces.
	@details This first find the face with the smallest unsigned square
	distance, and the returns the signed distance to that face (more efficient
	than the other version).
	@param[in] a_point 3D point
      */
      EBGEOMETRY_GPU_HOST_DEVICE
      [[nodiscard]] inline Real
      DirectSignedDistance2(const Vec3& a_point) const noexcept;

      /*!
	@brief Increment a warning. This is used in sanityCheck() for locating holes
	or bad inputs in the mesh.
	@param[in] a_warnings Map of all registered warnings
	@param[in] a_warn Current warning to increment by
      */
      EBGEOMETRY_GPU_HOST
      inline void
      incrementWarning(std::map<std::string, int>& a_warnings, const std::string& a_warn) const noexcept;

      /*!
	@brief Print all warnings to std::cerr
	@param[in] a_warnings All warnings
      */
      EBGEOMETRY_GPU_HOST
      inline void
      printWarnings(const std::map<std::string, int>& a_warnings) const noexcept;
    };
  } // namespace DCEL
} // namespace EBGeometry

#include "EBGeometry_DCEL_MeshImplem.hpp"

#endif
