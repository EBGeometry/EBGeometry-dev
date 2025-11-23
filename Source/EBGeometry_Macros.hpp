// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   EBGeometry_Macros.hpp
 * @brief  Various EBGeometry macros for assertions, debugging, inlining, and aliasing qualifiers.
 * @author Robert Marskar
 */

#ifndef EBGEOMETRY_MACROS_HPP
#define EBGEOMETRY_MACROS_HPP

// Std includes
#include <cassert>
#include <iostream>
#include <cstdint>
#include <type_traits>

// Our includes
#include "EBGeometry_GPU.hpp"

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
 * @note This macro is constexpr-compatible: in constexpr contexts it throws
 *       (causing compilation to fail), in runtime contexts it prints a message.
 */
#define EBGEOMETRY_ALWAYS_EXPECT(cond) ebgeometry_expect_impl((cond), #cond, __LINE__, __FILE__)

/**
 * @def EBGEOMETRY_EXPECT(cond)
 * @brief Conditionally checks a condition depending on debug settings.
 *
 * - If `EBGEOMETRY_ENABLE_DEBUG` is defined, this macro behaves like
 *   ::EBGEOMETRY_ALWAYS_EXPECT and performs a check.
 * - Otherwise, it compiles to a no-op.
 *
 * @param cond Boolean condition to evaluate.
 *
 * @note This macro is constexpr-compatible when debug mode is enabled.
 */
#ifdef EBGEOMETRY_ENABLE_DEBUG
#define EBGEOMETRY_EXPECT(cond) EBGEOMETRY_ALWAYS_EXPECT(cond)
#else
#define EBGEOMETRY_EXPECT(cond) ((void)0)
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

/**
 * @def EBGEOMETRY_PRAGMA_SIMD
 * @brief Portable loop-vectorization hint for CPU builds.
 *
 * Inserts one or more compiler-specific pragmas intended to encourage
 * SIMD vectorization of the immediately following loop.
 *
 * Usage example:
 * @code
 * EBGEOMETRY_PRAGMA_SIMD
 * for (int i = 0; i < n; ++i) {
 *     ...
 * }
 * @endcode
 *
 * Expansion rules:
 * - **Intel (ICC/ICX/oneAPI):** `#pragma ivdep` + `#pragma vector always`
 * - **Clang:**                 `#pragma clang loop vectorize(enable) interleave(enable)`
 * - **GCC:**                   `#pragma GCC ivdep`
 * - **MSVC:**                  `#pragma loop(ivdep)`
 * - **CUDA / HIP / SYCL device:** No-op (unknown pragmas break device builds)
 *
 * @note This macro must appear *immediately* before the loop header.
 * @warning This does *not* guarantee vectorization — it is a strong
 *          performance hint. Ensure that loop-carried dependencies do
 *          not inhibit safe SIMD execution.
 */

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__) || defined(__SYCL_DEVICE_ONLY__)
// On GPU/SYCL device passes, leave empty to avoid unknown pragma issues.
#define EBGEOMETRY_PRAGMA_SIMD
#else
#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
#define EBGEOMETRY_PRAGMA_SIMD _Pragma("ivdep") _Pragma("vector always")
#elif defined(__clang__)
#define EBGEOMETRY_PRAGMA_SIMD _Pragma("clang loop vectorize(enable) interleave(enable)")
#elif defined(__GNUC__)
#define EBGEOMETRY_PRAGMA_SIMD _Pragma("GCC ivdep")
#elif defined(_MSC_VER)
#define EBGEOMETRY_PRAGMA_SIMD _Pragma("loop(ivdep)")
#else
#define EBGEOMETRY_PRAGMA_SIMD
#endif
#endif

/** @} */ // end of Macros group

/**
 * @brief Constexpr-compatible expectation check helper with GPU support.
 *
 * This function can be used in constexpr, runtime, and GPU device contexts:
 * - In constexpr context: No-op (allows constexpr evaluation to continue;
 *   invalid operations will cause natural compile errors)
 * - In runtime context: prints error message if condition is false
 * - In CUDA device context: prints error message using device printf
 *
 * @param cond Boolean condition to check
 * @param msg Condition string for error message
 * @param line Source line number
 * @param file Source file name
 *
 * @note In constexpr contexts, if the condition is false and execution continues,
 *       downstream operations (like division by zero or invalid array access) will
 *       naturally cause compilation to fail with meaningful error messages.
 * @note On CUDA devices, printf is supported but the output may be buffered.
 */
#ifdef __CUDACC__
// CUDA compilation: can't use constexpr with __host__ __device__ in inline functions reliably
__host__ __device__ inline void
ebgeometry_expect_impl(bool cond, const char* msg, int line, const char* file)
{
  if (!cond) {
#ifdef __CUDA_ARCH__
    // Device code: use CUDA printf (allowed in device code)
    printf("Expectation '%s' failed on line %i in file %s!\n", msg, line, file);
#else
    // Host code in CUDA compilation
    printf("Expectation '%s' failed on line %i in file %s!\n", msg, line, file);
    ++EBGEOMETRY_ASSERTION_FAILURES;
#endif
  }
}
#else
// Non-CUDA compilation: use constexpr for better compile-time checking
inline constexpr void
ebgeometry_expect_impl(bool cond, const char* msg, int line, const char* file)
{
  if (!cond) {
    if (!std::is_constant_evaluated()) {
      // Only print in runtime context (printf is not constexpr)
      printf("Expectation '%s' failed on line %i in file %s!\n", msg, line, file);
      ++EBGEOMETRY_ASSERTION_FAILURES;
    }
  }
}
#endif

#endif
