// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   EBGeometry_TriangleCollection.hpp
 * @author Robert Marskar
 * @brief  Declaration of a triangle soup/collection (AoS/SoA) with signed distance functionality.
 *
 * @details This file provides two specialized implementations of TriangleCollection:
 *          - Array-of-Structs (AoS): Stores Triangle objects contiguously in memory
 *          - Struct-of-Arrays (SoA): Stores triangle components in separate arrays
 *
 *          Both specializations are lightweight, non-owning views suitable for GPU transfer.
 *          They implement signed distance queries to find the nearest triangle in the collection.
 *
 * @note All TriangleCollection specializations are non-owning. The caller is responsible
 *       for ensuring that the underlying data remains valid for the lifetime of the collection.
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
   * @brief Templated TriangleCollection declaration (primary).
   * @details This is the primary template declaration. Actual implementations are provided
   *          via full specializations for LayoutType::AoS and LayoutType::SoA.
   *
   * @tparam MetaData User-defined metadata type stored per triangle.
   * @tparam Layout   Memory layout type (LayoutType::AoS or LayoutType::SoA).
   */
  template <typename MetaData, LayoutType Layout>
  struct alignas(EBGEOMETRY_ALIGNAS) TriangleCollection;

  /**
   * @brief Array-of-Structs specialization of TriangleCollection.
   * @details Provides a lightweight, non-owning view over an array of Triangle objects.
   *          In AoS layout, all triangle data (vertices, normals, metadata) are stored
   *          contiguously within each Triangle object. This layout is cache-friendly for
   *          single-triangle access patterns and simpler to use when you already have
   *          an array of Triangle<MetaData> objects.
   *
   *          This type is trivially copyable, making it suitable for GPU kernel parameters.
   *
   * @tparam MetaData User-defined metadata type stored per triangle.
   *
   * Example usage:
   * @code
   * std::vector<Triangle<int>> triangles = {...};
   * TriangleCollection<int, LayoutType::AoS> collection(
   *     Span<const Triangle<int>>(triangles.data(), triangles.size())
   * );
   * Real distance = collection.value(query_point);
   * @endcode
   *
   * @note This is a non-owning view. The caller must ensure the underlying Triangle array
   *       remains valid for the lifetime of this collection.
   */
  template <typename MetaData>
  struct alignas(EBGEOMETRY_ALIGNAS) TriangleCollection<MetaData, LayoutType::AoS>
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
     * @brief Construct from a span of triangles.
     * @details Creates a non-owning view over the provided triangle array.
     * @param[in] a_triangles Span containing the triangle array to view.
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
     * @details Updates the internal span to point to a new triangle array.
     * @param[in] a_triangles Span containing the new triangle array to view (non-owning).
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
    length() const noexcept;

    /**
     * @brief Compute the signed distance from a point to the collection.
     * @details Returns the signed distance to the closest triangle (by absolute value).
     *          The sign follows the convention: negative inside, positive outside.
     * @param[in] a_point Query point in 3D space.
     * @return Signed distance to the nearest triangle. Returns EBGeometry::Limits::max()
     *         if the collection is empty (no triangles).
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Real
    value(const Vec3& a_point) const noexcept;

  protected:
    /**
     * @brief Non-owning span view of the triangle array.
     * @details Points to the underlying array of Triangle<MetaData> objects.
     *          This collection does not manage the lifetime of the data.
     */
    EBGeometry::Span<const Triangle<MetaData>> m_triangles{};
  };

  // Static assertions for AoS TriangleCollection
  static_assert(std::is_trivially_copyable_v<TriangleCollection<short, LayoutType::AoS>>,
                "TriangleCollection<AoS> must be trivially copyable for GPU compatibility");
  static_assert(std::is_trivially_copyable_v<TriangleCollection<int, LayoutType::AoS>>,
                "TriangleCollection<AoS> must be trivially copyable for GPU compatibility");
  static_assert(std::is_trivially_copyable_v<TriangleCollection<long, LayoutType::AoS>>,
                "TriangleCollection<AoS> must be trivially copyable for GPU compatibility");

  static_assert(std::is_standard_layout_v<TriangleCollection<short, LayoutType::AoS>>,
                "TriangleCollection<AoS> must have standard layout for GPU compatibility");
  static_assert(std::is_standard_layout_v<TriangleCollection<int, LayoutType::AoS>>,
                "TriangleCollection<AoS> must have standard layout for GPU compatibility");
  static_assert(std::is_standard_layout_v<TriangleCollection<long, LayoutType::AoS>>,
                "TriangleCollection<AoS> must have standard layout for GPU compatibility");

  /**
   * @brief Struct-of-Arrays specialization of TriangleCollection.
   * @details In SoA layout, triangle components are stored in separate arrays rather than
   *          grouped by triangle. This layout enables better SIMD vectorization across multiple
   *          triangles and can improve cache utilization for operations that process many
   *          triangles simultaneously.
   *
   *          This type is trivially copyable, making it suitable for GPU kernel parameters.
   *
   * @tparam MetaData User-defined metadata type stored per triangle.
   *
   * Memory Layout:
   *  - m_triangleNormals[i]       : triangle normal of triangle i
   *  - m_vertexPositions[k][i]    : vertex position k (0,1,2) of triangle i
   *  - m_vertexNormals[k][i]      : vertex normal k (0,1,2) of triangle i
   *  - m_edgeNormals[k][i]        : edge normal k (0,1,2) of triangle i
   *  - m_metadata[i]              : user metadata for triangle i
   *
   * Example usage:
   * @code
   * std::vector<Vec3> triangleNormals(N);
   * std::vector<Vec3> vertexPos0(N), vertexPos1(N), vertexPos2(N);
   * std::vector<Vec3> vertexNorm0(N), vertexNorm1(N), vertexNorm2(N);
   * std::vector<Vec3> edgeNorm0(N), edgeNorm1(N), edgeNorm2(N);
   * std::vector<int> metadata(N);
   *
   * TriangleCollection<int, LayoutType::SoA> collection(
   *     Span(triangleNormals), Span(vertexPos0), Span(vertexPos1), Span(vertexPos2),
   *     Span(vertexNorm0), Span(vertexNorm1), Span(vertexNorm2),
   *     Span(edgeNorm0), Span(edgeNorm1), Span(edgeNorm2), Span(metadata)
   * );
   * Real distance = collection.value(query_point);
   * @endcode
   *
   * @note All spans are non-owning views. The caller is responsible for ensuring that
   *       all underlying arrays remain valid for the lifetime of this collection.
   * @note All spans must have identical length equal to the number of triangles.
   */
  template <typename MetaData>
  struct alignas(EBGEOMETRY_ALIGNAS) TriangleCollection<MetaData, LayoutType::SoA>
  {
  public:
    /**
       * @brief Default constructor.
       * @details Creates an empty collection with null data pointers, zero triangles.
       */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr TriangleCollection() noexcept = default;

    /**
       * @brief Copy constructor.
       * @details Performs a shallow copy of all spans. Both collections will reference
       *          the same underlying data.
       * @param[in] a_other Other collection to copy from.
       */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr TriangleCollection(const TriangleCollection& a_other) noexcept = default;

    /**
       * @brief Move constructor.
       * @details Transfers ownership of span references from another collection.
       * @param[in,out] a_other Other collection to move from.
       */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr TriangleCollection(TriangleCollection&& a_other) noexcept = default;

    /**
     * @brief Construct from SoA spans.
     * @details All spans must have identical length equal to the number of triangles.
     *
     * @param[in] a_triangleNormals Triangle normal vectors (one per triangle).
     * @param[in] a_vertexPos0      Positions of vertex 0 for all triangles.
     * @param[in] a_vertexPos1      Positions of vertex 1 for all triangles.
     * @param[in] a_vertexPos2      Positions of vertex 2 for all triangles.
     * @param[in] a_vertexNorm0     Normals at vertex 0 for all triangles.
     * @param[in] a_vertexNorm1     Normals at vertex 1 for all triangles.
     * @param[in] a_vertexNorm2     Normals at vertex 2 for all triangles.
     * @param[in] a_edgeNorm0       Normals for edge 0 (between vertices 0 and 1) for all triangles.
     * @param[in] a_edgeNorm1       Normals for edge 1 (between vertices 1 and 2) for all triangles.
     * @param[in] a_edgeNorm2       Normals for edge 2 (between vertices 2 and 0) for all triangles.
     * @param[in] a_metadata        User metadata (one per triangle, must match triangle count).
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr TriangleCollection(EBGeometry::Span<const Vec3>     a_triangleNormals,
                                 EBGeometry::Span<const Vec3>     a_vertexPos0,
                                 EBGeometry::Span<const Vec3>     a_vertexPos1,
                                 EBGeometry::Span<const Vec3>     a_vertexPos2,
                                 EBGeometry::Span<const Vec3>     a_vertexNorm0,
                                 EBGeometry::Span<const Vec3>     a_vertexNorm1,
                                 EBGeometry::Span<const Vec3>     a_vertexNorm2,
                                 EBGeometry::Span<const Vec3>     a_edgeNorm0,
                                 EBGeometry::Span<const Vec3>     a_edgeNorm1,
                                 EBGeometry::Span<const Vec3>     a_edgeNorm2,
                                 EBGeometry::Span<const MetaData> a_metadata) noexcept;

    /**
       * @brief Copy assignment operator.
       * @details Performs a shallow copy of all spans. Both collections will reference
       *          the same underlying data.
       * @param[in] a_other Other collection to copy from.
       * @return Reference to this collection.
       */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr TriangleCollection&
    operator=(const TriangleCollection& a_other) noexcept = default;

    /**
       * @brief Move assignment operator.
       * @details Transfers ownership of span references from another collection.
       * @param[in,out] a_other Other collection to move from.
       * @return Reference to this collection.
       */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr TriangleCollection&
    operator=(TriangleCollection&& a_other) noexcept = default;

    /**
     * @brief Set SoA data.
     * @details Updates all internal spans to point to new data. All spans must have
     *          identical length equal to the number of triangles.
     *
     * @param[in] a_triangleNormals Triangle normal vectors (one per triangle).
     * @param[in] a_vertexPos0      Positions of vertex 0 for all triangles.
     * @param[in] a_vertexPos1      Positions of vertex 1 for all triangles.
     * @param[in] a_vertexPos2      Positions of vertex 2 for all triangles.
     * @param[in] a_vertexNorm0     Normals at vertex 0 for all triangles.
     * @param[in] a_vertexNorm1     Normals at vertex 1 for all triangles.
     * @param[in] a_vertexNorm2     Normals at vertex 2 for all triangles.
     * @param[in] a_edgeNorm0       Normals for edge 0 (between vertices 0 and 1) for all triangles.
     * @param[in] a_edgeNorm1       Normals for edge 1 (between vertices 1 and 2) for all triangles.
     * @param[in] a_edgeNorm2       Normals for edge 2 (between vertices 2 and 0) for all triangles.
     * @param[in] a_metadata        User metadata (one per triangle, must match triangle count).
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr void
    setData(EBGeometry::Span<const Vec3>     a_triangleNormals,
            EBGeometry::Span<const Vec3>     a_vertexPos0,
            EBGeometry::Span<const Vec3>     a_vertexPos1,
            EBGeometry::Span<const Vec3>     a_vertexPos2,
            EBGeometry::Span<const Vec3>     a_vertexNorm0,
            EBGeometry::Span<const Vec3>     a_vertexNorm1,
            EBGeometry::Span<const Vec3>     a_vertexNorm2,
            EBGeometry::Span<const Vec3>     a_edgeNorm0,
            EBGeometry::Span<const Vec3>     a_edgeNorm1,
            EBGeometry::Span<const Vec3>     a_edgeNorm2,
            EBGeometry::Span<const MetaData> a_metadata) noexcept;

    /**
     * @brief Get the number of triangles in the collection.
     * @return Number of triangles.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr int
    length() const noexcept
    {
      return m_numTriangles;
    }

    /**
     * @brief Compute signed distance from a point to the collection.
     * @details Returns the signed distance to the closest triangle (by absolute value).
     *          The sign follows the convention: negative inside, positive outside.
     *
     * @param[in] a_point Query point in 3D space.
     * @return Signed distance to the nearest triangle. Returns EBGeometry::Limits::max()
     *         if the collection is empty (no triangles).
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Real
    value(const Vec3& a_point) const noexcept;

  protected:
    /**
       * @brief Triangle normals (one per triangle).
       */
    EBGeometry::Span<const Vec3> m_triangleNormals{};

    /**
       * @brief Vertex positions for each triangle.
       * @details m_vertexPositions[k] contains positions for vertex k (0,1,2) across all triangles.
       */
    EBGeometry::Span<const Vec3> m_vertexPositions[3]{}; // [0],[1],[2]

    /**
       * @brief Vertex normals for each triangle.
       * @details m_vertexNormals[k] contains normals for vertex k (0,1,2) across all triangles.
       */
    EBGeometry::Span<const Vec3> m_vertexNormals[3]{}; // [0],[1],[2]

    /**
       * @brief Edge normals for each triangle.
       * @details m_edgeNormals[k] contains normals for edge k (0,1,2) across all triangles.
       *          Edge k is the edge between vertex k and vertex (k+1)%3.
       */
    EBGeometry::Span<const Vec3> m_edgeNormals[3]{}; // [0],[1],[2]

    /**
       * @brief User metadata (one per triangle).
       */
    EBGeometry::Span<const MetaData> m_metadata{};

    /**
       * @brief Number of triangles in the collection.
       */
    int m_numTriangles{0};
  };

  // Static assertions for SoA TriangleCollection
  static_assert(std::is_trivially_copyable_v<TriangleCollection<short, LayoutType::SoA>>,
                "TriangleCollection<SoA> must be trivially copyable for GPU compatibility");
  static_assert(std::is_trivially_copyable_v<TriangleCollection<int, LayoutType::SoA>>,
                "TriangleCollection<SoA> must be trivially copyable for GPU compatibility");
  static_assert(std::is_trivially_copyable_v<TriangleCollection<long, LayoutType::SoA>>,
                "TriangleCollection<SoA> must be trivially copyable for GPU compatibility");

  static_assert(std::is_standard_layout_v<TriangleCollection<short, LayoutType::SoA>>,
                "TriangleCollection<SoA> must have standard layout for GPU compatibility");
  static_assert(std::is_standard_layout_v<TriangleCollection<int, LayoutType::SoA>>,
                "TriangleCollection<SoA> must have standard layout for GPU compatibility");
  static_assert(std::is_standard_layout_v<TriangleCollection<long, LayoutType::SoA>>,
                "TriangleCollection<SoA> must have standard layout for GPU compatibility");

} // namespace EBGeometry

#include "EBGeometry_TriangleCollectionImplem.hpp"

#endif
