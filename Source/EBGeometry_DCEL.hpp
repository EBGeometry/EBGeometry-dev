// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   EBGeometry_DCEL.hpp
 * @brief  DCEL namespace with forward declarations.
 * @author Robert Marskar
 */

#ifndef EBGEOMETRY_DCEL_HPP
#define EBGEOMETRY_DCEL_HPP

namespace EBGeometry::DCEL {

  /**
   * @brief Enum for putting some logic into how vertex normal weights are calculated.
   */
  enum class VertexNormalWeight
  {
    None,
    Angle
  };

  /**
   * @brief Default meta-data type for the DCEL primitives
   */
  using DefaultMetaData = short;

  /**
   * @brief Vertex class
   */
  template <class Meta = DefaultMetaData>
  class Vertex;

  /**
   * @brief Edge class
   */
  template <class Meta = DefaultMetaData>
  class Edge;

  /**
   * @brief Face class
   */
  template <class Meta = DefaultMetaData>
  class Face;

  /**
   * @brief Mesh class
   */
  template <class Meta = DefaultMetaData>
  class Mesh;
} // namespace EBGeometry::DCEL

#endif
