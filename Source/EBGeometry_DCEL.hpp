/**
 * @file   EBGeometry_DCEL.hpp
 * @brief  DCEL namespace with forward declarations.
 * @author Robert Marskar
 */

#ifndef EBGeometry_DCEL
#define EBGeometry_DCEL

namespace EBGeometry::DCEL {

  /**
   * @brief Enum for putting some logic into how vertex normal weights are calculated.
   */
  enum class VertexNormalWeight // NOLINT
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
