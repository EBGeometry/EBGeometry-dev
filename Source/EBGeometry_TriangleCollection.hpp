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
    TriangleCollection() noexcept = default;

    /**
     * @brief Copy constructor.
     * @param[in] a_other Other collection to copy.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    TriangleCollection(const TriangleCollection& a_other) noexcept = default;

    /**
     * @brief Move constructor.
     * @param[in,out] a_other Other collection to move from.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    TriangleCollection(TriangleCollection&& a_other) noexcept = default;

    /**
     * @brief Construct from raw pointer and size.
     * @param[in] a_triangles Pointer to triangle array (non-owning).
     * @param[in] a_size Number of triangles in the array.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    TriangleCollection(const Triangle<MetaData>* a_triangles, int a_size) noexcept;

    /**
     * @brief Copy assignment.
     * @param[in] a_other Other collection to copy.
     * @return Reference to this collection.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    TriangleCollection&
    operator=(const TriangleCollection& a_other) noexcept = default;

    /**
     * @brief Move assignment.
     * @param[in,out] a_other Other collection to move from.
     * @return Reference to this collection.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    TriangleCollection&
    operator=(TriangleCollection&& a_other) noexcept = default;

    /**
     * @brief Set the collection data.
     * @param[in] a_triangles Pointer to triangle array (non-owning).
     * @param[in] a_size Number of triangles in the array.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    setData(const Triangle<MetaData>* a_triangles, int a_size) noexcept;

    /**
     * @brief Get the number of triangles in the collection.
     * @return Number of triangles.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    int
    size() const noexcept;

    /**
     * @brief Access triangle by index.
     * @param[in] i Index of triangle.
     * @return Reference to the triangle.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    const Triangle<MetaData>&
    operator[](int i) const noexcept;

    /**
     * @brief Compute the signed distance from a point to the collection.
     * @details Returns the signed distance to the closest triangle (by absolute value).
     * @param[in] a_point Query point.
     * @return Signed distance to the triangle soup.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    Real
    signedDistance(const Vec3& a_point) const noexcept;

  private:
    /**
     * @brief Pointer to triangle array (non-owning).
     */
    const Triangle<MetaData>* m_triangles = nullptr;

    /**
     * @brief Number of triangles in the collection.
     */
    int m_size = 0;
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
    TriangleCollection() noexcept = default;

    /**
     * @brief Copy constructor.
     * @param[in] a_other Other collection to copy.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    TriangleCollection(const TriangleCollection& a_other) noexcept = default;

    /**
     * @brief Move constructor.
     * @param[in,out] a_other Other collection to move from.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    TriangleCollection(TriangleCollection&& a_other) noexcept = default;

    /**
     * @brief Construct from raw arrays of triangle data.
     * @param[in] a_triangleNormals Pointer to array of triangle face normals.
     * @param[in] a_vertexPositions Array of pointers to vertex position arrays (3 arrays).
     * @param[in] a_vertexNormals Array of pointers to vertex normal arrays (3 arrays).
     * @param[in] a_edgeNormals Array of pointers to edge normal arrays (3 arrays).
     * @param[in] a_metaData Pointer to array of metadata objects.
     * @param[in] a_size Number of triangles.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    TriangleCollection(const Vec3*     a_triangleNormals,
                       const Vec3*     a_vertexPositions[3],
                       const Vec3*     a_vertexNormals[3],
                       const Vec3*     a_edgeNormals[3],
                       const MetaData* a_metaData,
                       int             a_size) noexcept;

    /**
     * @brief Copy assignment.
     * @param[in] a_other Other collection to copy.
     * @return Reference to this collection.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    TriangleCollection&
    operator=(const TriangleCollection& a_other) noexcept = default;

    /**
     * @brief Move assignment.
     * @param[in,out] a_other Other collection to move from.
     * @return Reference to this collection.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    TriangleCollection&
    operator=(TriangleCollection&& a_other) noexcept = default;

    /**
     * @brief Set the collection data.
     * @param[in] a_triangleNormals Pointer to array of triangle face normals.
     * @param[in] a_vertexPositions Array of pointers to vertex position arrays (3 arrays).
     * @param[in] a_vertexNormals Array of pointers to vertex normal arrays (3 arrays).
     * @param[in] a_edgeNormals Array of pointers to edge normal arrays (3 arrays).
     * @param[in] a_metaData Pointer to array of metadata objects.
     * @param[in] a_size Number of triangles.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    void
    setData(const Vec3*     a_triangleNormals,
            const Vec3*     a_vertexPositions[3],
            const Vec3*     a_vertexNormals[3],
            const Vec3*     a_edgeNormals[3],
            const MetaData* a_metaData,
            int             a_size) noexcept;

    /**
     * @brief Get the number of triangles in the collection.
     * @return Number of triangles.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    int
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
    signedDistance(const Vec3& a_point) const noexcept;

  protected:
    /**
     * @brief Compute signed distance to a single triangle given its SoA fields.
     * @param[in] a_triangleNormal Face normal of triangle.
     * @param[in] a_vertexPositions Array of vertex positions (3).
     * @param[in] a_vertexNormals Array of vertex normals (3).
     * @param[in] a_edgeNormals Array of edge normals (3).
     * @param[in] a_point Query point.
     * @return Signed distance to triangle.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    static Real
    signedDistanceTriangle(const Vec3& a_triangleNormal,
                           const Vec3  a_vertexPositions[3],
                           const Vec3  a_vertexNormals[3],
                           const Vec3  a_edgeNormals[3],
                           const Vec3& a_point) noexcept;

  private:
    /**
       @brief Pointer to array of triangle face normals.
    */
    const Vec3* m_triangleNormal = nullptr;

    /**
       @brief Pointers to arrays of vertex positions (3 arrays, one per vertex).
    */
    const Vec3* m_vertexPositions[3]{nullptr, nullptr, nullptr};

    /**
       @brief Pointers to arrays of vertex normals (3 arrays, one per vertex).
    */
    const Vec3* m_vertexNormals[3]{nullptr, nullptr, nullptr};

    /**
       @brief Pointers to arrays of edge normals (3 arrays, one per edge).
    */
    const Vec3* m_edgeNormals[3]{nullptr, nullptr, nullptr};

    /**
       @brief Pointer to array of per-triangle metadata.
    */
    const MetaData* m_metaData = nullptr;

    /**
       @brief Number of triangles in the collection.
    */
    int m_size = 0;
  };

} // namespace EBGeometry

#include "EBGeometry_TriangleCollectionImplem.hpp"

#endif // EBGeometry_TriangleCollection
