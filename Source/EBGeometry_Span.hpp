// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   EBGeometry_Span.hpp
 * @brief  Declaration of a lightweight GPU-safe Span view.
 * @author Robert Marskar
 *
 * @details
 * A minimal, non-owning, contiguous view over a block of memory, similar in spirit
 * to `std::span` but designed to be usable in GPU device code. The span stores a
 * pointer and a size (in elements). Bounds checking is the caller's responsibility.
 *
 * This type is trivially copyable and intended for POD-like usage across host and device.
 */

#ifndef EBGEOMETRY_SPAN_HPP
#define EBGEOMETRY_SPAN_HPP

// Std includes
#include <cstddef>
#include <type_traits>

// Our includes
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_Macros.hpp"

namespace EBGeometry {

  /**
   * @brief Lightweight GPU-safe span that references a contiguous sequence of `T` without owning it.
   * @tparam T Element type (const-correctness is respected by the pointer type).
   *
   * @note This type mirrors a strict subset of `std::span` functionality:
   *   - default constructor (empty span),
   *   - pointer + length constructor,
   *   - element access via operator[],
   *   - `length()`, `begin()`, and `end()` accessors.
   *
   * All member functions are annotated for host and device execution.
   */
  template <typename T>
  struct Span
  {
    /**
     * @brief Raw pointer to the first element (non-owning, may be nullptr for empty spans).
     */
    const T* m_data = nullptr;

    /**
     * @brief Number of elements in the span.
     */
    int m_size = 0;

    /**
     * @brief Default constructor. Constructs an empty span.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr Span() noexcept;

    /**
     * @brief Pointer + size constructor.
     * @param[in] a_data Pointer to the first element in the sequence (non-owning).
     * @param[in] a_size Number of elements in the sequence.
     * @warning Caller is responsible for ensuring the memory region `[a_data, a_data + s)` is valid.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    EBGEOMETRY_ALWAYS_INLINE
    constexpr Span(const T* a_data, int a_size) noexcept;

    /**
     * @brief Element access (no bounds checking).
     * @param[in] i Index in the range `[0, size)`.
     * @return Const reference to the i-th element.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr const T&
    operator[](int i) const noexcept;

    /**
     * @brief Number of elements in the span.
     * @return Number of elements.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr int
    length() const noexcept;

    /**
     * @brief Iterator to the first element.
     * @return Pointer to first element.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr const T*
    begin() const noexcept;

    /**
     * @brief Iterator one past the last element.
     * @return Pointer to one past the last element.
     */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    constexpr const T*
    end() const noexcept;
  };

  // Basic guarantees for POD-like usage on host/device.
  static_assert(std::is_trivially_copyable<Span<int>>::value, "EBGeometry::Span must be trivially copyable");
  static_assert(std::is_standard_layout<Span<int>>::value, "EBGeometry::Span must be standard layout");
} // namespace EBGeometry

#include "EBGeometry_SpanImplem.hpp"

#endif
