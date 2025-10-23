// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   EBGeometry_TriangleCollection.hpp
 * @author Robert Marskar
 * @brief  Declaration of a triangle soup collection (AoS/SoA) with signed distance functionality.
 */

#ifndef EBGeometry_TriangleCollection
#define EBGeometry_TriangleCollection

// Our includes
#include "EBGeometry_Alignas.hpp"
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_GPUTypes.hpp"
#include "EBGeometry_Macros.hpp"
#include "EBGeometry_Vec.hpp"
#include "EBGeometry_Span.hpp"
#include "EBGeometry_Triangle.hpp"

namespace EBGeometry {

  /**
   * @brief Templated TriangleCollection declaration (primary).
   * @tparam MetaData  User metadata stored per triangle.
   * @tparam Layout    Layout type (AoS or SoA).
   */
  template <typename MetaData, LayoutType Layout>
  class alignas(EBGEOMETRY_ALIGNAS) TriangleCollection;

  /**
   * @brief Array-of-Structs specialization of TriangleCollection.
   * @details Provides a lightweight, non-owning view over an array of Triangle objects.
   *          Suited for simple usage or when you already store `Triangle<MetaData>` items.
   *
   * @tparam MetaData Metadata type stored per triangle.
   */
  template <typename MetaData>
  class alignas(EBGEOMETRY_ALIGNAS) TriangleCollection<MetaData, LayoutType::AoS>
  {
  public:
    /**
     * @brief Default constructor.
     * @details Creates an empty collection with null data pointer and zero size.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr TriangleCollection() noexcept = default;

    /**
     * @brief Copy constructor.
     * @param[in] a_other Other collection to copy.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr TriangleCollection(const TriangleCollection& a_other) noexcept = default;

    /**
     * @brief Move constructor.
     * @param[in,out] a_other Other collection to move from.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr TriangleCollection(TriangleCollection&& a_other) noexcept = default;

    /**
     * @brief Construct from raw pointer and size.
     * @param[in] a_triangles List of triangles.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr explicit TriangleCollection(EBGeometry::Span<const Triangle<MetaData>> a_triangles) noexcept;

    /**
     * @brief Copy assignment.
     * @param[in] a_other Other collection to copy.
     * @return Reference to this collection.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr TriangleCollection&
    operator=(const TriangleCollection& a_other) noexcept = default;

    /**
     * @brief Move assignment.
     * @param[in,out] a_other Other collection to move from.
     * @return Reference to this collection.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr TriangleCollection&
    operator=(TriangleCollection&& a_other) noexcept = default;

    /**
     * @brief Set the collection data.
     * @param[in] a_triangles Pointer to triangle array (non-owning).
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr void
    setData(EBGeometry::Span<const Triangle<MetaData>> a_triangles) noexcept;

    /**
     * @brief Get the number of triangles in the collection.
     * @return Number of triangles.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr int
    size() const noexcept;

    /**
     * @brief Compute the signed distance from a point to the collection.
     * @details Returns the signed distance to the closest triangle (by absolute value).
     * @param[in] a_point Query point.  
     * @return Signed distance to the triangle soup.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Real
    value(const Vec3& a_point) const noexcept;

  protected:
    /**
     * @brief Pointer to triangle array (non-owning).
     */
    EBGeometry::Span<const Triangle<MetaData>> m_triangles{};
  };

  /**
   * @brief Structure-of-Arrays specialization of TriangleCollection.
   * @details Provides a lightweight, non-owning view over arrays of triangle attributes.
   *          All arrays must have the same length and be aligned/sized appropriately.
   *
   * @tparam MetaData Metadata type stored per triangle.
   */
  template <typename MetaData>
  class alignas(EBGEOMETRY_ALIGNAS) TriangleCollection<MetaData, LayoutType::SoA>
  {
  public:
    /**
     * @brief Default constructor.
     * @details Creates an empty collection with null pointers and zero size.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr TriangleCollection() noexcept = default;

    /**
     * @brief Copy constructor.
     * @param[in] a_other Other collection to copy.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr TriangleCollection(const TriangleCollection& a_other) noexcept = default;

    /**
     * @brief Move constructor.
     * @param[in,out] a_other Other collection to move from.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr TriangleCollection(TriangleCollection&& a_other) noexcept = default;

    /**
     * @brief Set the SoA triangle data.
     * @param[in] a_triangleNormals Sequence of triangle face normals.
     * @param[in] a_firstVertexPositions First triangle vertex positions.
     * @param[in] a_secondVertexPositions Second triangle vertex positions.
     * @param[in] a_thirdVertexPositions Third triangle vertex positions.
     * @param[in] a_firstVertexNormals First triangle vertex normals.
     * @param[in] a_secondVertexNormals Second triangle vertex normals.
     * @param[in] a_thirdVertexNormals Third triangle vertex normals.
     * @param[in] a_firstEdgeNormals First triangle edge normals.
     * @param[in] a_secondEdgeNormals Second triangle edge normals.
     * @param[in] a_thirdEdgeNormals Third triangle edge normals.     
     * @param[in] a_metaData Sequence of metadata objects for each triangle.       
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr explicit TriangleCollection(EBGeometry::Span<const Vec3>     a_triangleNormals,
                                          EBGeometry::Span<const Vec3>     a_firstVertexPositions,
                                          EBGeometry::Span<const Vec3>     a_secondVertexPositions,
                                          EBGeometry::Span<const Vec3>     a_thirdVertexPositions,
                                          EBGeometry::Span<const Vec3>     a_firstVertexNormals,
                                          EBGeometry::Span<const Vec3>     a_secondVertexNormals,
                                          EBGeometry::Span<const Vec3>     a_thirdVertexNormals,
                                          EBGeometry::Span<const Vec3>     a_firstEdgeNormals,
                                          EBGeometry::Span<const Vec3>     a_secondEdgeNormals,
                                          EBGeometry::Span<const Vec3>     a_thirdEdgeNormals,
                                          EBGeometry::Span<const MetaData> a_metaData) noexcept;
    /**
     * @brief Copy assignment.
     * @param[in] a_other Other collection to copy.
     * @return Reference to this collection.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr TriangleCollection&
    operator=(const TriangleCollection& a_other) noexcept = default;

    /**
     * @brief Move assignment.
     * @param[in,out] a_other Other collection to move from.
     * @return Reference to this collection.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr TriangleCollection&
    operator=(TriangleCollection&& a_other) noexcept = default;

    /**
     * @brief Set the SoA triangle data.
     * @param[in] a_triangleNormals Sequence of triangle face normals.
     * @param[in] a_firstVertexPositions First triangle vertex positions.
     * @param[in] a_secondVertexPositions Second triangle vertex positions.
     * @param[in] a_thirdVertexPositions Third triangle vertex positions.
     * @param[in] a_firstVertexNormals First triangle vertex normals.
     * @param[in] a_secondVertexNormals Second triangle vertex normals.
     * @param[in] a_thirdVertexNormals Third triangle vertex normals.
     * @param[in] a_firstEdgeNormals First triangle edge normals.
     * @param[in] a_secondEdgeNormals Second triangle edge normals.
     * @param[in] a_thirdEdgeNormals Third triangle edge normals.     
     * @param[in] a_metaData Sequence of metadata objects for each triangle.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr void
    setData(EBGeometry::Span<const Vec3>     a_triangleNormals,
            EBGeometry::Span<const Vec3>     a_firstVertexPositions,
            EBGeometry::Span<const Vec3>     a_secondVertexPositions,
            EBGeometry::Span<const Vec3>     a_thirdVertexPositions,
            EBGeometry::Span<const Vec3>     a_firstVertexNormals,
            EBGeometry::Span<const Vec3>     a_secondVertexNormals,
            EBGeometry::Span<const Vec3>     a_thirdVertexNormals,
            EBGeometry::Span<const Vec3>     a_firstEdgeNormals,
            EBGeometry::Span<const Vec3>     a_secondEdgeNormals,
            EBGeometry::Span<const Vec3>     a_thirdEdgeNormals,
            EBGeometry::Span<const MetaData> a_metaData) noexcept;

    /**
     * @brief Get the number of triangles in the collection.
     * @return Number of triangles.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr int
    size() const noexcept;

    /**
     * @brief Compute the signed distance from a point to the collection.
     * @details Returns the signed distance to the closest triangle (by absolute value).
     * @param[in] a_point Query point.
     * @return Signed distance to the triangle soup.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Real
    value(const Vec3& a_point) const noexcept;

  protected:
    /**
       @brief Sequence of triangle face normals. 
    */
    EBGeometry::Span<const Vec3> m_triangleNormals{};

    /**
       @brief Sequence of triangle vertex positions.
    */
    EBGeometry::Span<const Vec3> m_firstVertexPositions{};

    /**
       @brief Sequence of triangle vertex positions.
    */
    EBGeometry::Span<const Vec3> m_secondVertexPositions{};

    /**
       @brief Sequence of triangle vertex positions.
    */
    EBGeometry::Span<const Vec3> m_thirdVertexPositions{};

    /**
       @brief Sequence of triangle vertex normals on the first vertex in each triangle. 
    */
    EBGeometry::Span<const Vec3> m_firstVertexNormals{};

    /**
       @brief Sequence of triangle vertex normals on the second vertex in each triangle.        
    */
    EBGeometry::Span<const Vec3> m_secondVertexNormals{};

    /**
       @brief Sequence of triangle vertex normals on the third vertex in each triangle.               
    */
    EBGeometry::Span<const Vec3> m_thirdVertexNormals{};

    /**
       @brief Sequence of triangle edge normals.
    */
    EBGeometry::Span<const Vec3> m_firstEdgeNormals{};

    /**
       @brief Sequence of triangle edge normals.
    */
    EBGeometry::Span<const Vec3> m_secondEdgeNormals{};

    /**
       @brief Sequence of triangle edge normals.
    */
    EBGeometry::Span<const Vec3> m_thirdEdgeNormals{};

    /**
       @brief Sequence of triangle metadata.
    */
    EBGeometry::Span<const MetaData> m_metaData{};

    /**
     * @brief Compute signed distance to a single triangle given its SoA fields.
     * @param[in] a_triangleNormal Face normal of triangle.
     * @param[in] a_vertexPositions Array of vertex positions (length 3).
     * @param[in] a_vertexNormals Array of vertex normals (length 3).
     * @param[in] a_edgeNormals Array of edge normals (length 3).
     * @param[in] a_point Query point.
     * @return Signed distance to triangle.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    static Real
    signedDistanceTriangle(const Vec3&                     a_triangleNormal,
                           const Vec3* EBGEOMETRY_RESTRICT a_vertexPositions,
                           const Vec3* EBGEOMETRY_RESTRICT a_vertexNormals,
                           const Vec3* EBGEOMETRY_RESTRICT a_edgeNormals,
                           const Vec3&                     a_point) noexcept;
  };
} // namespace EBGeometry

#include "EBGeometry_TriangleCollectionImplem.hpp"

#endif // EBGeometry_TriangleCollection
