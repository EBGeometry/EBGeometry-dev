// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   EBGeometry_TriangleCollection.hpp
 * @author Robert Marskar
 * @brief  Declaration of a triangle soup/collection (AoS/SoA) with signed distance functionality.
 */

#ifndef EBGEOMETRY_TRIANGLECOLLECTION_HPP
#define EBGEOMETRY_TRIANGLECOLLECTION_HPP

// Our includes
#include "EBGeometry_Alignas.hpp"
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_GPUTypes.hpp"
#include "EBGeometry_LayoutType.hpp"
#include "EBGeometry_Macros.hpp"
#include "EBGeometry_Span.hpp"
#include "EBGeometry_Triangle.hpp"
#include "EBGeometry_Vec.hpp"

namespace EBGeometry {

  /**
   * @brief Primary template for a collection of triangles with signed distance queries.
   *
   * This primary template is specialized for different memory layouts:
   *  - `LayoutType::AoS` : Array-of-Structs, storing `Triangle<MetaData>` elements.
   *  - `LayoutType::SoA` : Struct-of-Arrays, storing separate arrays for triangle data.
   *
   * @tparam MetaData User-defined metadata stored per triangle.
   * @tparam Layout   Memory layout (AoS or SoA).
   */
  template <typename MetaData, LayoutType Layout>
  struct alignas(EBGEOMETRY_ALIGNAS) TriangleCollection;

  //=====================================================================
  // AoS specialization
  //=====================================================================

  /**
   * @brief Array-of-Structs specialization of TriangleCollection.
   *
   * This specialization holds a non-owning span over an array of `Triangle<MetaData>`
   * instances. It provides a convenient view and a signed-distance query function.
   *
   * @tparam MetaData User-defined metadata stored per triangle.
   */
  template <typename MetaData>
  struct alignas(EBGEOMETRY_ALIGNAS) TriangleCollection<MetaData, LayoutType::AoS>
  {
  public:
    /**
     * @brief Default constructor.
     *
     * Creates an empty collection: @c m_triangles is an empty span and
     * @c length() returns 0.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr TriangleCollection() noexcept = default;

    /**
     * @brief Construct from a span of Triangle objects.
     *
     * The collection becomes a non-owning view over the given triangle span.
     *
     * @param[in] a_triangles Span over `Triangle<MetaData>` objects.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr explicit TriangleCollection(EBGeometry::Span<const Triangle<MetaData>> a_triangles) noexcept;

    /**
     * @brief Set the triangle data for this collection.
     *
     * The collection becomes a non-owning view over the given triangle span.
     *
     * @param[in] a_triangles Span over `Triangle<MetaData>` objects.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr void
    setData(EBGeometry::Span<const Triangle<MetaData>> a_triangles) noexcept;

    /**
     * @brief Get the number of triangles in the collection.
     *
     * @return The number of triangles referenced by the collection.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr int
    length() const noexcept;

    /**
     * @brief Compute the signed distance from a point to this triangle collection.
     *
     * The function finds the triangle in the collection that yields the
     * smallest absolute signed distance to the query point and returns that
     * signed distance. If the collection is empty, a sentinel value such as
     * `EBGeometry::Limits::max` is returned (depending on DistanceCandidate
     * initialization).
     *
     * @param[in] a_point Query point in 3D space.
     * @return Signed distance to the closest triangle (by absolute value).
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Real
    value(const Vec3& a_point) const noexcept;

  protected:
    /**
     * @brief Non-owning span over the underlying triangle array.
     *
     * This span references externally stored `Triangle<MetaData>` objects.
     * The collection does not manage the memory lifetime of these objects.
     */
    EBGeometry::Span<const Triangle<MetaData>> m_triangles{};
  };

  //=====================================================================
  // SoA specialization
  //=====================================================================

  /**
   * @brief Struct-of-Arrays specialization of TriangleCollection.
   *
   * This specialization uses parallel arrays (SoA layout) to store per-triangle
   * data, which can improve memory access patterns and SIMD utilization.
   *
   * Each triangle @c i is represented by:
   *  - @c m_tn[i]  : Triangle normal.
   *  - @c m_vx1[i] : Position of vertex 1.
   *  - @c m_vx2[i] : Position of vertex 2.
   *  - @c m_vx3[i] : Position of vertex 3.
   *  - @c m_vn1[i] : Normal at vertex 1.
   *  - @c m_vn2[i] : Normal at vertex 2.
   *  - @c m_vn3[i] : Normal at vertex 3.
   *  - @c m_en1[i] : Edge-1 normal.
   *  - @c m_en2[i] : Edge-2 normal.
   *  - @c m_en3[i] : Edge-3 normal.
   *  - @c m_metadata[i] : Optional user metadata associated with the triangle.
   *
   * All spans are non-owning views into externally managed memory.
   *
   * @tparam MetaData User-defined metadata stored per triangle.
   */
  template <typename MetaData>
  struct alignas(EBGEOMETRY_ALIGNAS) TriangleCollection<MetaData, LayoutType::SoA>
  {
  public:
    /**
     * @brief Default constructor.
     *
     * Creates an empty collection where all spans are empty, and
     * @c m_numTriangles is set to zero.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr TriangleCollection() noexcept = default;

    /**
     * @brief Construct from SoA spans for all triangle data.
     *
     * All spans are assumed to have a consistent length (number of triangles).
     * The metadata span @c a_metadata is optional; when empty, the collection
     * will consider metadata as unused.
     *
     * @param[in] a_tn       Triangle normals span. Element @c i is the normal of triangle @c i.
     * @param[in] a_vx1      Vertex-1 positions span. Element @c i is vertex 1 of triangle @c i.
     * @param[in] a_vx2      Vertex-2 positions span. Element @c i is vertex 2 of triangle @c i.
     * @param[in] a_vx3      Vertex-3 positions span. Element @c i is vertex 3 of triangle @c i.
     * @param[in] a_vn1      Vertex-1 normals span. Element @c i is the normal at vertex 1 of triangle @c i.
     * @param[in] a_vn2      Vertex-2 normals span. Element @c i is the normal at vertex 2 of triangle @c i.
     * @param[in] a_vn3      Vertex-3 normals span. Element @c i is the normal at vertex 3 of triangle @c i.
     * @param[in] a_en1      Edge-1 normals span for each triangle.
     * @param[in] a_en2      Edge-2 normals span for each triangle.
     * @param[in] a_en3      Edge-3 normals span for each triangle.
     * @param[in] a_metadata Optional metadata span; element @c i corresponds to triangle @c i.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr TriangleCollection(EBGeometry::Span<const Vec3>     a_tn,
                                 EBGeometry::Span<const Vec3>     a_vx1,
                                 EBGeometry::Span<const Vec3>     a_vx2,
                                 EBGeometry::Span<const Vec3>     a_vx3,
                                 EBGeometry::Span<const Vec3>     a_vn1,
                                 EBGeometry::Span<const Vec3>     a_vn2,
                                 EBGeometry::Span<const Vec3>     a_vn3,
                                 EBGeometry::Span<const Vec3>     a_en1,
                                 EBGeometry::Span<const Vec3>     a_en2,
                                 EBGeometry::Span<const Vec3>     a_en3,
                                 EBGeometry::Span<const MetaData> a_metadata = {}) noexcept;

    /**
     * @brief Set the SoA data for this triangle collection.
     *
     * All spans are assumed to have consistent length (number of triangles).
     * The metadata span @c a_metadata is optional; when empty, the collection
     * will consider metadata as unused.
     *
     * @param[in] a_tn       Triangle normals span. Element @c i is the normal of triangle @c i.
     * @param[in] a_vx1      Vertex-1 positions span. Element @c i is vertex 1 of triangle @c i.
     * @param[in] a_vx2      Vertex-2 positions span. Element @c i is vertex 2 of triangle @c i.
     * @param[in] a_vx3      Vertex-3 positions span. Element @c i is vertex 3 of triangle @c i.
     * @param[in] a_vn1      Vertex-1 normals span. Element @c i is the normal at vertex 1 of triangle @c i.
     * @param[in] a_vn2      Vertex-2 normals span. Element @c i is the normal at vertex 2 of triangle @c i.
     * @param[in] a_vn3      Vertex-3 normals span. Element @c i is the normal at vertex 3 of triangle @c i.
     * @param[in] a_en1      Edge-1 normals span for each triangle.
     * @param[in] a_en2      Edge-2 normals span for each triangle.
     * @param[in] a_en3      Edge-3 normals span for each triangle.
     * @param[in] a_metadata Optional metadata span; element @c i corresponds to triangle @c i.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr void setData(EBGeometry::Span<const Vec3>     a_tn,
                           EBGeometry::Span<const Vec3>     a_vx1,
                           EBGeometry::Span<const Vec3>     a_vx2,
                           EBGeometry::Span<const Vec3>     a_vx3,
                           EBGeometry::Span<const Vec3>     a_vn1,
                           EBGeometry::Span<const Vec3>     a_vn2,
                           EBGeometry::Span<const Vec3>     a_vn3,
                           EBGeometry::Span<const Vec3>     a_en1,
                           EBGeometry::Span<const Vec3>     a_en2,
                           EBGeometry::Span<const Vec3>     a_en3,
                           EBGeometry::Span<const MetaData> a_metadata = {}) noexcept;

    /**
     * @brief Get the number of triangles in the collection.
     *
     * This corresponds to the length of @c m_tn and the other geometry spans.
     *
     * @return The number of triangles referenced by the SoA collection.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr int
    length() const noexcept { return m_numTriangles; }

    /**
     * @brief Compute the signed distance from a point to this triangle collection.
     *
     * The function finds the triangle in the collection that yields the
     * smallest absolute signed distance to the query point and returns that
     * signed distance. If the collection is empty, a sentinel value such as
     * `EBGeometry::Limits::max` is returned (depending on DistanceCandidate
     * initialization).
     *
     * @param[in] a_point Query point in 3D space.
     * @return Signed distance to the closest triangle (by absolute value).
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Real
    value(const Vec3& a_point) const noexcept;

  protected:
    /**
     * @brief Span of triangle normals.
     *
     * Element @c i is the triangle normal of triangle @c i.
     */
    EBGeometry::Span<const Vec3> m_tn{};

    /**
     * @brief Span of vertex-1 positions.
     *
     * Element @c i is the position of vertex 1 of triangle @c i.
     */
    EBGeometry::Span<const Vec3> m_vx1{};

    /**
     * @brief Span of vertex-2 positions.
     *
     * Element @c i is the position of vertex 2 of triangle @c i.
     */
    EBGeometry::Span<const Vec3> m_vx2{};

    /**
     * @brief Span of vertex-3 positions.
     *
     * Element @c i is the position of vertex 3 of triangle @c i.
     */
    EBGeometry::Span<const Vec3> m_vx3{};

    /**
     * @brief Span of vertex-1 normals.
     *
     * Element @c i is the normal at vertex 1 of triangle @c i.
     */
    EBGeometry::Span<const Vec3> m_vn1{};

    /**
     * @brief Span of vertex-2 normals.
     *
     * Element @c i is the normal at vertex 2 of triangle @c i.
     */
    EBGeometry::Span<const Vec3> m_vn2{};

    /**
     * @brief Span of vertex-3 normals.
     *
     * Element @c i is the normal at vertex 3 of triangle @c i.
     */
    EBGeometry::Span<const Vec3> m_vn3{};

    /**
     * @brief Span of edge-1 normals for each triangle.
     *
     * Element @c i is the edge-1 normal for triangle @c i.
     */
    EBGeometry::Span<const Vec3> m_en1{};

    /**
     * @brief Span of edge-2 normals for each triangle.
     *
     * Element @c i is the edge-2 normal for triangle @c i.
     */
    EBGeometry::Span<const Vec3> m_en2{};

    /**
     * @brief Span of edge-3 normals for each triangle.
     *
     * Element @c i is the edge-3 normal for triangle @c i.
     */
    EBGeometry::Span<const Vec3> m_en3{};

    /**
     * @brief Span of per-triangle metadata.
     *
     * Element @c i is the metadata associated with triangle @c i.
     * This span may be empty if metadata is not used.
     */
    EBGeometry::Span<const MetaData> m_metadata{};

    /**
     * @brief Number of triangles in the collection.
     *
     * This is typically equal to the length of @c m_tn and the other geometry spans.
     */
    int m_numTriangles{0};
  };

} // namespace EBGeometry

#include "EBGeometry_TriangleCollectionImplem.hpp"

#endif // EBGEOMETRY_TRIANGLECOLLECTION_HPP
