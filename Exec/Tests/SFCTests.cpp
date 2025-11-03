// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "EBGeometry.hpp"

#include <cstdint>
#include <vector>
#include <tuple>
#include <limits>
#include <type_traits>

#include <catch2/catch_test_macros.hpp>

using namespace EBGeometry::SFC;

TEST_CASE("Index: basic properties & construction", "[sfc][index]")
{
  // Type assumptions
  STATIC_REQUIRE(std::is_trivially_copyable<Index>::value);
  STATIC_REQUIRE(std::is_standard_layout<Index>::value);
  STATIC_REQUIRE(std::is_same_v<IntType, uint32_t>);

  // Size/layout: exactly 3 * 4 bytes (no padding on common ABIs)
  STATIC_REQUIRE(sizeof(Index) == 3 * sizeof(IntType));

  SECTION("Default-initialized values are zero")
  {
    Index a{0, 0, 0};
    REQUIRE(a.x == 0);
    REQUIRE(a.y == 0);
    REQUIRE(a.z == 0);
  }

  SECTION("Construct with typical values")
  {
    Index a{1, 2, 3};
    REQUIRE(a.x == 1);
    REQUIRE(a.y == 2);
    REQUIRE(a.z == 3);
  }

  SECTION("Construct at allowed bounds")
  {
    const IntType max = static_cast<IntType>(ValidSpan);
    Index         a{max, max, max}; // inclusive bound
    REQUIRE(a.x == max);
    REQUIRE(a.y == max);
    REQUIRE(a.z == max);
  }
}

TEST_CASE("Morton: known small vectors", "[sfc][morton][known]")
{
  // Hand-checked 3D Morton codes:
  // bit lanes: x -> bits 0,3,6,... ; y -> bits 1,4,7,... ; z -> bits 2,5,8,...
  struct KV
  {
    Index p;
    Code  code;
  };

  const std::vector<KV> cases = {
    {{0, 0, 0}, 0ull},
    {{1, 0, 0}, 1ull},  // x0
    {{0, 1, 0}, 2ull},  // y0
    {{0, 0, 1}, 4ull},  // z0
    {{1, 1, 0}, 3ull},  // 1|2
    {{1, 0, 1}, 5ull},  // 1|4
    {{0, 1, 1}, 6ull},  // 2|4
    {{1, 1, 1}, 7ull},  // 1|2|4
    {{2, 0, 0}, 8ull},  // x1 -> bit3
    {{0, 2, 0}, 16ull}, // y1 -> bit4
    {{0, 0, 2}, 32ull}, // z1 -> bit5
    {{3, 0, 0}, 9ull},  // x0|x1 -> bits0&3 = 1+8
    {{0, 3, 0}, 18ull}, // y0|y1 -> bits1&4 = 2+16
    {{0, 0, 3}, 36ull}, // z0|z1 -> bits2&5 = 4+32
    {{5, 0, 0}, 65ull}, // x0|x2 -> bits0&6 = 1+64
  };

  for (const auto& kv : cases) {
    const auto got = Morton::encode(kv.p);
    REQUIRE(got == kv.code);

    const auto back = Morton::decode(kv.code);
    REQUIRE(back.x == kv.p.x);
    REQUIRE(back.y == kv.p.y);
    REQUIRE(back.z == kv.p.z);
  }
}

TEST_CASE("Morton: reversibility on a small grid", "[sfc][morton][roundtrip]")
{
  // Exercise a decent chunk but keep runtime low
  const IntType max = 63; // small grid [0..63]^3
  for (IntType x = 0; x <= max; ++x) {
    for (IntType y = 0; y <= max; ++y) {
      for (IntType z = 0; z <= max; ++z) {
        Index       p{x, y, z};
        const Code  c = Morton::encode(p);
        const Index q = Morton::decode(c);
        REQUIRE(q.x == x);
        REQUIRE(q.y == y);
        REQUIRE(q.z == z);
      }
    }
  }
}

TEST_CASE("Morton: reversibility at edges", "[sfc][morton][edges]")
{
  const IntType max = static_cast<IntType>(ValidSpan);

  // Some edge combos, including the extreme corner
  const std::vector<Index> points =
    {{0, 0, 0}, {max, 0, 0}, {0, max, 0}, {0, 0, max}, {max, max, 0}, {max, 0, max}, {0, max, max}, {max, max, max}};

  for (const auto& p : points) {
    const Code  c = Morton::encode(p);
    const Index q = Morton::decode(c);
    REQUIRE(q.x == p.x);
    REQUIRE(q.y == p.y);
    REQUIRE(q.z == p.z);
  }
}

TEST_CASE("Nested: known vectors and reversibility", "[sfc][nested]")
{
  const Code N  = static_cast<Code>(ValidSpan) + 1;
  const Code N2 = N * N;

  auto linearize = [&](Index p) -> Code { return static_cast<Code>(p.x) + static_cast<Code>(p.y) * N + static_cast<Code>(p.z) * N2; };

  SECTION("Known small vectors")
  {
    struct KV
    {
      Index p;
      Code  code;
    };
    const std::vector<KV> cases = {
      {{0, 0, 0}, linearize({0, 0, 0})},
      {{1, 0, 0}, linearize({1, 0, 0})},
      {{0, 1, 0}, linearize({0, 1, 0})},
      {{0, 0, 1}, linearize({0, 0, 1})},
      {{1, 2, 3}, linearize({1, 2, 3})},
      {{5, 4, 3}, linearize({5, 4, 3})},
    };

    for (const auto& kv : cases) {
      const auto got = Nested::encode(kv.p);
      REQUIRE(got == kv.code);

      const auto back = Nested::decode(kv.code);
      REQUIRE(back.x == kv.p.x);
      REQUIRE(back.y == kv.p.y);
      REQUIRE(back.z == kv.p.z);
    }
  }

  SECTION("Round-trip on a small grid")
  {
    const IntType max = 63; // small grid [0..63]^3
    for (IntType x = 0; x <= max; ++x) {
      for (IntType y = 0; y <= max; ++y) {
        for (IntType z = 0; z <= max; ++z) {
          Index       p{x, y, z};
          const Code  c = Nested::encode(p);
          const Index q = Nested::decode(c);
          REQUIRE(q.x == x);
          REQUIRE(q.y == y);
          REQUIRE(q.z == z);
        }
      }
    }
  }

  SECTION("Edges, including the extreme corner")
  {
    const IntType            max = static_cast<IntType>(ValidSpan);
    const std::vector<Index> points =
      {{0, 0, 0}, {max, 0, 0}, {0, max, 0}, {0, 0, max}, {max, max, 0}, {max, 0, max}, {0, max, max}, {max, max, max}};

    for (const auto& p : points) {
      const Code  c = Nested::encode(p);
      const Index q = Nested::decode(c);
      REQUIRE(q.x == p.x);
      REQUIRE(q.y == p.y);
      REQUIRE(q.z == p.z);
    }

    // Also check that max code fits in 64-bit and matches N^3 - 1
    const Code maxCode = Nested::encode({max, max, max});
    REQUIRE(maxCode == (N * N * N - 1));
    // Sanity: N^3 - 1 == 2^63 - 1 for ValidBits=21 (fits in uint64_t)
    STATIC_REQUIRE(sizeof(Code) == 8);
    REQUIRE(maxCode <= std::numeric_limits<Code>::max());
  }
}
