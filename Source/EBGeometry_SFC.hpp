// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   EBGeometry_SFC.hpp
 * @brief  Declaration of various space-filling curves
 * @author Robert Marskar
 */

#ifndef EBGEOMETRY_SFC_HPP
#define EBGEOMETRY_SFC_HPP

// Std includes
#include <cstdint>
#include <type_traits>

// Our includes
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_Macros.hpp"

/**
 * @brief Namespace for holding space-filling curve functionality.
 * @details The SFC code is always a 64-bit integer (this can be changed, but bit-shifts have to be redone in that case).
 */
namespace EBGeometry::SFC {

  /**
   * @brief Alias for SFC code
   */
  using Code = uint64_t;

  /**
   * @brief Alias for coordinate width
   */
  using IntType = uint32_t;

  /**
   * @brief Maximum available bits.
   *
   * Note: Morton implementation below assumes ValidBits <= 21 (3*21 = 63 bits).
   */
  static constexpr unsigned int ValidBits = 21;

  /**
   * @brief Maximum permitted span along any spatial coordinate (inclusive).
   * Valid coordinates are in [0, ValidSpan].
   */
  static constexpr uint64_t ValidSpan = (static_cast<uint64_t>(1) << ValidBits) - 1;

  /**
   * @brief Simple POD index for usage with SFC codes.
   */
  struct Index
  {
    /**
     * @brief Constructor — sets x, y, and z
     * @param[in] a_x First coordinate
     * @param[in] a_y Second coordinate
     * @param[in] a_z Third coordinate
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    Index(uint32_t a_x, uint32_t a_y, uint32_t a_z) noexcept;

    /**
     * @brief First index
     */
    uint32_t x{0};

    /**
     * @brief Second index
     */
    uint32_t y{0};

    /**
     * @brief Third index
     */
    uint32_t z{0};
  };

  /**
   * @brief Encodable SFC concept -- class must have static encode/decode:
   *        static uint64_t encode(const Index&);
   *        static Index    decode(const uint64_t&);
   */
  template <typename S>
  concept Encodable = requires(const SFC::Index& point, const SFC::Code code) {
    { S::encode(point) } -> std::same_as<SFC::Code>;
    { S::decode(code) } -> std::same_as<SFC::Index>;
  };

  /**
   * @brief Implementation of the Morton SFC (3D)
   */
  struct Morton
  {
    /**
     * @brief Helper function for interleaving every third bit.
     * @param[in] v Input number coordinate.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    static uint64_t
    interleaveBits21(uint32_t v) noexcept;

    /**
     * @brief Helper function for compacting every third bit.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    static uint32_t
    compactBits21(uint64_t x) noexcept;

    /**
     * @brief Encode an input point into a Morton index with a 64-bit representation.
     * @param[in] a_point Input point.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    static Code
    encode(const Index& a_point) noexcept;

    /**
     * @brief Decode the 64-bit Morton code into an Index.
     * @param[in] a_code SFC code
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    static Index
    decode(const Code& a_code) noexcept;
  };

  /**
   * @brief Implementation of a nested index SFC.
   * @details The SFC is encoded by code = i + j * N + k * N * N in 3D,
   *          where N = ValidSpan + 1 is the extent per axis.
   */
  struct Nested
  {
    /**
     * @brief Encoding function.
     * @param[in] a_point Input point to turn into a SFC code
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    static uint64_t
    encode(const Index& a_point) noexcept;

    /**
     * @brief Decoding function.
     * @param[in] a_code SFC code to be turned into coordinate.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    static Index
    decode(const uint64_t& a_code) noexcept;
  };
} // namespace EBGeometry::SFC

static_assert(std::is_trivially_copyable<EBGeometry::SFC::Index>::value, "EBGeometry::SFC::Index must be trivially copyable");
static_assert(std::is_standard_layout<EBGeometry::SFC::Index>::value, "EBGeometry::SFC::Index must have a standard layout");

static_assert(std::is_trivially_copyable<EBGeometry::SFC::Morton>::value, "EBGeometry::SFC::Morton must be trivially copyable");
static_assert(std::is_standard_layout<EBGeometry::SFC::Morton>::value, "EBGeometry::SFC::Morton must have standard layout");
static_assert(EBGeometry::SFC::Encodable<EBGeometry::SFC::Morton>, "EBGeometry::SFC::Morton must fulfill encodable concept");

static_assert(std::is_trivially_copyable<EBGeometry::SFC::Nested>::value, "EBGeometry::SFC::Nested must be trivially copyable");
static_assert(std::is_standard_layout<EBGeometry::SFC::Nested>::value, "EBGeometry::SFC::Nested must have standard layout");
static_assert(EBGeometry::SFC::Encodable<EBGeometry::SFC::Nested>, "EBGeometry::SFC::Nested must fulfill encodable concept");

#include "EBGeometry_SFCImplem.hpp"

#endif
