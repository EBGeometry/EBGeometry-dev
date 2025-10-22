// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   EBGeometry_Macros.hpp
 * @brief  Various EBGeometry macros for assertions, debugging, inlining, and aliasing qualifiers.
 * @author Robert Marskar
 */

#ifndef EBGeometry_Macros
#define EBGeometry_Macros

#include <cassert>
#include <iostream>

/**
 * @brief Global counter tracking the number of assertion failures.
 *
 * This variable increments whenever an assertion failure is detected
 * in EBGeometry. It can be used for debugging or testing purposes
 * to monitor how often expectations are violated.
 */
static unsigned long long int EBGEOMETRY_ASSERTION_FAILURES = 0;

/**
 * @defgroup Macros EBGeometry Macros
 * @brief Collection of utility macros for EBGeometry.
 *
 * This group contains assertion, debugging, inlining, and aliasing macros
 * that provide consistent behavior across compilers and build modes.
 * @{
 */

/**
 * @def EBGEOMETRY_ALWAYS_EXPECT(cond)
 * @brief Always checks a condition and reports failure if it is false.
 *
 * This macro performs an unconditional expectation check, regardless
 * of whether debugging is enabled. If @p cond evaluates to false,
 * it prints an error message including the failed condition, source
 * line, and file.
 *
 * @param cond Boolean condition to evaluate.
 *
 * @note Unlike assertions, this does not abort the program.
 */
#define EBGEOMETRY_ALWAYS_EXPECT(cond)                                                     \
  if (!(cond)) {                                                                           \
    printf("Expectation '%s' failed on line %i in file %s!\n", #cond, __LINE__, __FILE__); \
  }

/**
 * @def EBGEOMETRY_EXPECT(cond)
 * @brief Conditionally checks a condition depending on debug settings.
 *
 * - If `EBGEOMETRY_ENABLE_DEBUG` is defined, this macro behaves like
 *   ::EBGEOMETRY_ALWAYS_EXPECT and performs a runtime check.
 * - Otherwise, it compiles to a no-op.
 *
 * @param cond Boolean condition to evaluate.
 */
#ifdef EBGEOMETRY_ENABLE_DEBUG
#define EBGEOMETRY_EXPECT(cond) EBGEOMETRY_ALWAYS_EXPECT(cond)
#else
#define EBGEOMETRY_EXPECT(cond) (void)0
#endif

/**
 * @def EBGEOMETRY_ALWAYS_INLINE
 * @brief Macro for enforcing function inlining across different compilers.
 *
 * Expands to:
 * - `__forceinline__` if compiling with CUDA (`__CUDA_ARCH__`) and
 *   `EBGEOMETRY_ENABLE_CUDA` is defined.
 * - `inline __attribute__((always_inline))` for GCC.
 * - `inline` for all other compilers.
 *
 * This is useful for performance-critical functions that must be inlined
 * regardless of compiler optimization heuristics.
 */
#if defined(__CUDA_ARCH__) && defined(EBGEOMETRY_ENABLE_CUDA)
#define EBGEOMETRY_ALWAYS_INLINE __forceinline__

#elif defined(__GNUC__)
#define EBGEOMETRY_ALWAYS_INLINE inline __attribute__((always_inline))

#else
#define EBGEOMETRY_ALWAYS_INLINE inline
#endif

/**
 * @def EBGEOMETRY_INLINE
 * @brief Generic inline macro, expands to `inline`.
 *
 * This macro provides a uniform way to mark functions as inline without
 * forcing compiler-specific attributes.
 */
#define EBGEOMETRY_INLINE inline

/**
 * @def EBGEOMETRY_RESTRICT
 * @brief Portable aliasing qualifier that maps to the compiler's restrict extension.
 *
 * Many compilers support a non-standard `restrict`-like qualifier that enables
 * more aggressive optimizations by promising non-aliasing pointers/references.
 * This macro expands to:
 * - `__restrict__` for Clang and GCC.
 * - `__restrict`  for MSVC and Intel compilers (ICC, ICX, oneAPI).
 * - Empty definition on unknown compilers (no-op).
 *
 * @warning Only use ::EBGEOMETRY_RESTRICT when you can *guarantee* that two or more
 *          qualified pointers/references do not alias the same memory region.
 *          Violating this promise yields undefined behavior.
 *
 * @note Typical usage is to qualify local raw pointers extracted from views/spans
 *       inside hot loops (e.g., signed-distance sweeps) to improve auto-vectorization.
 */
#if defined(__clang__) || defined(__GNUC__)
#define EBGEOMETRY_RESTRICT __restrict__
#elif defined(_MSC_VER) || defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
#define EBGEOMETRY_RESTRICT __restrict
#else
#define EBGEOMETRY_RESTRICT
#endif

/** @} */ // end of Macros group

#endif
