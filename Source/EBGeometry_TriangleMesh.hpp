/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file    EBGeometry_MeshDistanceFunctions.hpp
  @brief   Declaration of various mesh-based distance functions.
  @author  Robert Marskar
*/

#ifndef EBGeometry_TriangleMesh
#define EBGeometry_TriangleMesh

#include "EBGeometry_Types.hpp"
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_GPUTypes.hpp"
#include "EBGeometry_ImplicitFunction.hpp"
#include "EBGeometry_Macros.hpp"
#include "EBGeometry_Triangle.hpp"
#include "EBGeometry_Vec.hpp"

namespace EBGeometry {

  /*!
    @brief Triangle mesh for usage as implicit function.
    @details This class has a signed distance functionality, but only if the following conditions are fulfilled:

    1. The mesh is watertight.
    2. The mesh is orientable, without flipped faces.
    3. The mesh is manifold without self-intersections.
    
    It is up to the user to ensure that these conditions are fulfilled. If they are, then this class can be used
    as an implicit function. 
  */
  template <typename MetaData>
  class TriangleMesh : public ImplicitFunction
  {
  public:
    /*
      @brief Default constructor. Does not initialize any triangles
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    TriangleMesh() noexcept = default;

    /*
      @brief Full constructor. Initializes the triangle mesh.
      @param[in] a_numTriangles Number of triangles in the mesh.
      @param[in] a_triangles List of triangles. 
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    TriangleMesh(long long int a_numTriangles, const Triangle<MetaData>* a_triangles) noexcept;

    /*!
      @brief Copy constructor. 
      @param[in] a_triangleMesh The triangle mesh to copy.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    TriangleMesh(const TriangleMesh& a_mesh) noexcept = default;

    /*!
      @brief Move constructor. 
      @param[in] a_triangleMesh The triangle mesh to move.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    TriangleMesh(TriangleMesh&& a_mesh) noexcept = default;

    /*!
      @brief Destructor -- does not deallocate memory. 
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    ~TriangleMesh() noexcept = default;

    /*
      @brief Define function which associates the triangles. 
      @param[in] a_numTriangles Number of triangles in the mesh.
      @param[in] a_triangles List of triangles. 
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    define(long long int a_numTriangles, const Triangle<MetaData>* a_triangles) noexcept;

    /*!
      @brief Get the number of triangles in the mesh
      @return Returns m_numTriangles
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    long long int
    getNumberOfTriangles() const noexcept;

    /*!
      @brief Get the triangles in the mesh
      @return Returns m_triangles
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    const Triangle<MetaData>* 
    getTriangles() const noexcept;    

    /*!
      @brief Value function. Returns the (signed) distance to the closest triangle
      @param[in] a_point Query point. 
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Real
    value(const Vec3& a_point) const noexcept override;

  protected:
    /*!
      @brief List of triangles
    */
    const Triangle<MetaData>* m_triangles = nullptr;

    /*!
      @brief Number of triangles
    */
    long long int m_numTriangles = -1LL;
  };
} // namespace EBGeometry

#include "EBGeometry_TriangleMeshImplem.hpp" // NOLINT

#endif
