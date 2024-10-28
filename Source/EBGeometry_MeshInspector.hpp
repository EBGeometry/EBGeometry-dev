/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_MeshInspector.hpp
  @brief  Utility functions for inspecting and modifying polygon soups. 
  @author Robert Marskar
*/

#ifndef EBGeometry_MeshInspector
#define EBGeometry_MeshInspector

// Std includes
#include <vector>

// Our includes
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_Vec.hpp"

namespace EBGeometry {

  /*!
    @brief Namespace for parsing files into various distance functions representations. None of these functions
    are callabled on the device.
  */
  namespace MeshInspector {

    /*!
      @brief Check if polygons in a polygon soup contain degenerate vertices
      @param[out] a_vertices Vertices
      @param[out] a_polygons Polygons
    */
    EBGEOMETRY_GPU_HOST
    EBGEOMETRY_ALWAYS_INLINE static bool
    containsDegeneratePolygons(const std::vector<EBGeometry::Vec3>& a_vertices,
                               const std::vector<std::vector<int>>& a_polygons) noexcept;

    /*!
      @brief Compress polygon soup. This removes degenerate polygons (e.g., triangles).
      @details This will iterate through 
      @param[in, out] a_vertices Vertices
      @param[in, out] a_polygons Planar polygons.
    */
    EBGEOMETRY_GPU_HOST
    EBGEOMETRY_ALWAYS_INLINE static void
    removeDegeneratePolygonsFromSoup(std::vector<EBGeometry::Vec3>& a_vertices,
                                     std::vector<std::vector<int>>& a_polygons) noexcept;

  } // namespace MeshInspector
} // namespace EBGeometry

#include "EBGeometry_MeshParserImplem.hpp"

#endif
