/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_DCEL.hpp
  @brief  Declaration of useful utilities for the DCEL functionality.
  @author Robert Marskar
*/

#ifndef EBGeometry_DCEL
#define EBGeometry_DCEL

namespace EBGeometry {

  /*!
    @namespace DCEL
    @brief Namespace containing various double-connected edge list (DCEL)
    functionality.
  */
  namespace DCEL {

    /*!
      @brief Default meta-data type for DCEL primitives.
      @details This is a primitive-attached type used for attaching properties to DCEL vertices, edges, and
      faces. Very useful if one wants to pass boundary conditions through the DCEL primitives.
    */
    using DefaultMetaData = short;

    /*!
      @brief DCEL vertex class
    */
    template <class Meta = DefaultMetaData>
    class Vertex;

    /*!
      @brief DCEL half-edge class
    */
    template <class Meta = DefaultMetaData>
    class Edge;

    /*!
      @brief DCEL face class
    */
    template <class Meta = DefaultMetaData>
    class Face;

    /*!
      @brief DCEL mesh class
    */
    template <class Meta = DefaultMetaData>
    class Mesh;

    /*!
      @brief DCEL edge iterator class
    */
    template <class Meta = DefaultMetaData>
    class EdgeIterator;

    /*!
      @brief Enum for putting some logic into how vertex normal weights are calculated.
    */
    enum class VertexNormalWeight
    {
      None,
      Angle
    };

  } // namespace DCEL
} // namespace EBGeometry

#include "EBGeometry_DCEL_VertexImplem.hpp"

#endif
