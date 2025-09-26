/**
 * @file   EBGeometry_SFCImplem.hpp
 * @brief  Implementation of EBGeometry_SFC.hpp
 * @author Robert Marskar
 */

#ifndef EBGeometry_SFCImplem
#define EBGeometry_SFCImplem

// Std includes
#include <climits>

// Our includes
#include "EBGeometry_SFC.hpp"
#include "EBGeometry_Macros.hpp"

namespace EBGeometry::SFC {

  EBGEOMETRY_ALWAYS_INLINE
  Index::Index(unsigned int x, unsigned int y, unsigned int z) noexcept :
    m_indices{x, y, z}
  {}

  EBGEOMETRY_ALWAYS_INLINE
  unsigned int
  Index::operator[](int a_dir) const noexcept
  {
    EBGEOMETRY_EXPECT(a_dir <= 2);
    EBGEOMETRY_EXPECT(a_dir >= 0);

    return m_indices[a_dir];
  }

  EBGEOMETRY_ALWAYS_INLINE
  uint64_t
  Morton::encode(const Index& a_point) noexcept
  {
    uint64_t code = 0;

    const auto x = static_cast<uint_fast32_t>(a_point[0]);
    const auto y = static_cast<uint_fast32_t>(a_point[1]);
    const auto z = static_cast<uint_fast32_t>(a_point[2]);

    auto PartBy3 = [](const uint_fast32_t a) -> uint64_t {
      uint64_t b = a & Mask_64[0];

      b = (b | b << 32) & Mask_64[1];
      b = (b | b << 16) & Mask_64[2];
      b = (b | b << 8) & Mask_64[3];
      b = (b | b << 4) & Mask_64[4];
      b = (b | b << 2) & Mask_64[5];

      return b;
    };

    code |= PartBy3(x) | PartBy3(y) << 1 | PartBy3(z) << 2;

    return code;
  }

  EBGEOMETRY_ALWAYS_INLINE
  Index
  Morton::decode(const uint64_t& a_code) noexcept
  {
    auto getEveryThirdBit = [](const uint64_t m) -> uint_fast32_t {
      uint_fast32_t x = m & Mask_64[5];

      x = (x ^ (x >> 2)) & Mask_64[4];
      x = (x ^ (x >> 4)) & Mask_64[3];
      x = (x ^ (x >> 8)) & Mask_64[2];
      x = (x ^ (x >> 16)) & Mask_64[1];
      x = (x ^ (x >> 32)) & Mask_64[0];

      return x;
    };

    const auto x = static_cast<unsigned int>(getEveryThirdBit(a_code));
    const auto y = static_cast<unsigned int>(getEveryThirdBit(a_code >> 1));
    const auto z = static_cast<unsigned int>(getEveryThirdBit(a_code >> 2));

    return Index(x, y, z);
  }

  EBGEOMETRY_ALWAYS_INLINE
  uint64_t
  Nested::encode(const Index& a_point) noexcept
  {
    uint64_t code = 0;

    const uint32_t x = a_point[0];
    const uint32_t y = a_point[1];
    const uint32_t z = a_point[2];

    code = x + (y * SFC::ValidSpan) + (z * SFC::ValidSpan * SFC::ValidSpan);

    return code;
  }

  EBGEOMETRY_ALWAYS_INLINE
  Index
  Nested::decode(const uint64_t& a_code) noexcept
  {
    const unsigned int z = a_code / (SFC::ValidSpan * SFC::ValidSpan);
    const unsigned int y = (a_code - z * SFC::ValidSpan * SFC::ValidSpan) / SFC::ValidSpan;
    const unsigned int x = a_code - z * SFC::ValidSpan * SFC::ValidSpan - y * SFC::ValidSpan;

    return Index(x, y, z);
  }
} // namespace EBGeometry::SFC

#endif
