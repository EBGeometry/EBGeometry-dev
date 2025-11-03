// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   EBGeometry_SFCImplem.hpp
 * @brief  Implementation of EBGeometry_SFC.hpp
 * @author Robert Marskar
 */

#ifndef EBGEOMETRY_SFCIMPLEM_HPP
#define EBGEOMETRY_SFCIMPLEM_HPP

// Std includes
#include <climits>
#include <cstdint>

// Our includes
#include "EBGeometry_SFC.hpp"
#include "EBGeometry_Macros.hpp"

namespace EBGeometry::SFC {

  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  Index::Index(uint32_t a_x, uint32_t a_y, uint32_t a_z) noexcept :
    x(a_x),
    y(a_y),
    z(a_z)
  {
    EBGEOMETRY_EXPECT(a_x <= static_cast<uint32_t>(ValidSpan));
    EBGEOMETRY_EXPECT(a_y <= static_cast<uint32_t>(ValidSpan));
    EBGEOMETRY_EXPECT(a_z <= static_cast<uint32_t>(ValidSpan));
  }

  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  uint64_t
  Morton::interleaveBits21(uint32_t v) noexcept
  {
    uint64_t x = static_cast<uint64_t>(v & 0x1fffffU);

    x = (x | (x << 32)) & 0x1F00000000FFFFULL;
    x = (x | (x << 16)) & 0x1F0000FF0000FFULL;
    x = (x | (x << 8)) & 0x100F00F00F00F00FULL;
    x = (x | (x << 4)) & 0x10C30C30C30C30C3ULL;
    x = (x | (x << 2)) & 0x1249249249249249ULL;

    return x;
  }

  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  uint32_t
  Morton::compactBits21(uint64_t x) noexcept
  {
    x &= 0x1249249249249249ULL;

    x = (x ^ (x >> 2)) & 0x10C30C30C30C30C3ULL;
    x = (x ^ (x >> 4)) & 0x100F00F00F00F00FULL;
    x = (x ^ (x >> 8)) & 0x1F0000FF0000FFULL;
    x = (x ^ (x >> 16)) & 0x1F00000000FFFFULL;
    x = (x ^ (x >> 32)) & 0x00000000001FFFFFULL;

    return static_cast<uint32_t>(x);
  }

  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  uint64_t
  Morton::encode(const Index& a_point) noexcept
  {
    EBGEOMETRY_EXPECT(a_point.x <= static_cast<uint32_t>(ValidSpan));
    EBGEOMETRY_EXPECT(a_point.y <= static_cast<uint32_t>(ValidSpan));
    EBGEOMETRY_EXPECT(a_point.z <= static_cast<uint32_t>(ValidSpan));

    const uint64_t xlane = interleaveBits21(a_point.x);
    const uint64_t ylane = interleaveBits21(a_point.y) << 1;
    const uint64_t zlane = interleaveBits21(a_point.z) << 2;

    return xlane | ylane | zlane;
  }

  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  Index
  Morton::decode(const uint64_t& a_code) noexcept
  {
    const uint32_t x = compactBits21(a_code);
    const uint32_t y = compactBits21(a_code >> 1);
    const uint32_t z = compactBits21(a_code >> 2);

    return Index{x, y, z};
  }

  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  uint64_t
  Nested::encode(const Index& a_point) noexcept
  {
    EBGEOMETRY_EXPECT(a_point.x <= static_cast<uint32_t>(ValidSpan));
    EBGEOMETRY_EXPECT(a_point.y <= static_cast<uint32_t>(ValidSpan));
    EBGEOMETRY_EXPECT(a_point.z <= static_cast<uint32_t>(ValidSpan));

    const uint64_t N = static_cast<uint64_t>(ValidSpan) + 1;

    return static_cast<uint64_t>(a_point.x) + static_cast<uint64_t>(a_point.y) * N + static_cast<uint64_t>(a_point.z) * N * N;
  }

  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  Index
  Nested::decode(const uint64_t& a_code) noexcept
  {
    const uint64_t N  = static_cast<uint64_t>(ValidSpan) + 1;
    const uint64_t N2 = N * N;

    const uint64_t zc = a_code / N2;
    const uint64_t yc = (a_code - zc * N2) / N;
    const uint64_t xc = a_code - zc * N2 - yc * N;

    return Index{static_cast<uint32_t>(xc), static_cast<uint32_t>(yc), static_cast<uint32_t>(zc)};
  }
} // namespace EBGeometry::SFC

#endif // EBGEOMETRY_SFCIMPLEM_HPP
