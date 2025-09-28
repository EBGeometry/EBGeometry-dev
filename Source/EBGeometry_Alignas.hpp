// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   EBGeometry_Alignas.hpp
 * @brief  Declaration of alignment sizes for various architectures
 * @author Robert Marskar
 */

#ifndef EBGEOMETRY_ALIGNAS_HPP
#define EBGEOMETRY_ALIGNAS_HPP

// Std includes
#include <cstddef>

/**
 * @brief Alignment size used for EBGeometry types.
 *
 * @details This constant expands to the natural alignment required by the
 * highest available SIMD instruction set at compile time:
 * - **64 bytes** if AVX-512 is available
 * - **32 bytes** if AVX/AVX2 is available
 * - **16 bytes** if SSE2, ARM NEON, or WebAssembly SIMD128 is available
 * - Otherwise defaults to `alignof(std::max_align_t)`
 *
 * The value is chosen to ensure safe and efficient memory alignment
 * for vectorized operations on the target architecture.
 */
#if defined(__AVX512F__)
inline constexpr std::size_t EBGEOMETRY_ALIGNAS = 64;
#elif defined(__AVX2__) || defined(__AVX__)
inline constexpr std::size_t EBGEOMETRY_ALIGNAS = 32;
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
inline constexpr std::size_t EBGEOMETRY_ALIGNAS = 16;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
inline constexpr std::size_t EBGEOMETRY_ALIGNAS = 16;
#elif defined(__wasm_simd128__)
inline constexpr std::size_t EBGEOMETRY_ALIGNAS = 16;
#else
inline constexpr std::size_t EBGEOMETRY_ALIGNAS = alignof(std::max_align_t);
#endif

#endif
