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

#warning "SoA version is missing -- needs to get written ASAP"

namespace EBGeometry {

  /**
   * @brief Templated TriangleCollection declaration (primary).
   * @tparam MetaData  User metadata stored per triangle.
   * @tparam Layout    Layout type (AoS or SoA).
   */
  template <typename MetaData, LayoutType Layout>
  struct alignas(EBGEOMETRY_ALIGNAS) TriangleCollection;

  /**
   * @brief Array-of-Structs specialization of TriangleCollection.
   * @details Provides a lightweight, non-owning view over an array of Triangle objects.
   *          Suited for simple usage or when you already store `Triangle<MetaData>` items.
   *
   * @tparam MetaData Metadata type stored per triangle.
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
    length() const noexcept;

    /**
     * @brief Compute the signed distance from a point to the collection.
     * @details Returns the signed distance to the closest triangle (by absolute value).
     * @param[in] a_point Query point.  
     * @return Signed distance to the triangle collection. Returns EBGeometry::Limits::max if there are no triangles     
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
} // namespace EBGeometry

#include "EBGeometry_TriangleCollectionImplem.hpp"

#endif
