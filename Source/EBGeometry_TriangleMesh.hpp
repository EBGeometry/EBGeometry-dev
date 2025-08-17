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
    @brief Triangle mesh for usage as implicit function.
    @details This class has a signed distance functionality, but only if the following conditions are fulfilled:

    1. The mesh is watertight.
    2. The mesh is orientable, without flipped faces.
    3. The mesh is manifold without self-intersections.
    
    It is up to the user to ensure that these conditions are fulfilled. If they are, then this class can be used
    as an implicit function. 
  */
  template <typename MetaData>
  class TriangleMesh
  {
  public:

    /*
      @brief Default constructor. Does not initialize any triangles
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMENTRY_ALWAYS_INLINE
    TriangleMesh() noexcept = default;

  protected:
    
  };
} // namespace EBGeometry

#endif
