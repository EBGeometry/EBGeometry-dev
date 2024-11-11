/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file    EBGeometry_MeshDistanceFunctions.hpp
  @brief   Declaration of various mesh-based distance functions.
  @author  Robert Marskar
*/

#ifndef EBGeometry_MeshDistanceFunctions
#define EBGeometry_MeshDistanceFunctions

#include "EBGeometry_Types.hpp"
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_GPUTypes.hpp"
#include "EBGeometry_ImplicitFunction.hpp"
#include "EBGeometry_Macros.hpp"
#include "EBGeometry_Vec.hpp"

namespace EBGeometry {

  /*!
    @brief Signed distance function for a DCEL mesh.
    @details This version does not use the BVH accelerator.
  */
  template <class MetaData = DCEL::DefaultMetaData>
  class DCELMeshSDF : public ImplicitFunction
  {
  public:
    /*!
      @brief Full constructor. Puts object in usable state.
      @details This class is non-owning, and the base mesh only resides as a pointer.
      @param[in] a_mesh DCEL mesh. Must be manifold for distance field to make sense.
      @note If constructing on the GPU, the input mesh must also live on the GPU. Usually, this 
      constructor will be called through a factory method that cares of this for you. 
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    DCELMeshSDF(const EBGeometry::DCEL::Mesh<MetaData>* const a_mesh) noexcept
    {
      EBGEOMETRY_EXPECT(a_mesh != nullptr);
      EBGEOMETRY_EXPECT(a_mesh->isManifold());

      this->m_mesh = a_mesh;
    }

    /*!
      @brief Destructor (does nothing)
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    ~DCELMeshSDF() noexcept
    {}

    /*!
      @brief Signed distance function (or value function) for the DCEL mesh
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Real
    value(const Vec3& a_point) const noexcept override
    {
      EBGEOMETRY_EXPECT(m_mesh != nullptr);

      return m_mesh->signedDistance(a_point);
    }

  protected:
    /*!
      @brief Reference to the mesh. 
    */
    const DCEL::Mesh<MetaData>* m_mesh;
  };
} // namespace EBGeometry

#endif
