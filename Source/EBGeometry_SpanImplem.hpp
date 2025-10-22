// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   EBGeometry_SpanImplem.hpp
 * @brief  Implementation of EBGeometry_Span.hpp
 * @author Robert Marskar
 */

#ifndef EBGEOMETRY_SPANIMPLEM_HPP
#define EBGEOMETRY_SPANIMPLEM_HPP

// Our includes
#include "EBGeometry_Span.hpp"
#include "EBGeometry_GPUTypes.hpp"
#include "EBGeometry_Macros.hpp"

namespace EBGeometry {

  template <typename T>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  constexpr Span<T>::Span() noexcept :
    m_data(nullptr),
    m_size(0)
  {}

  template <typename T>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  constexpr Span<T>::Span(const T* a_data, int a_size) noexcept :
    m_data(a_data),
    m_size(a_size)
  {}

  template <typename T>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  constexpr const T&
  Span<T>::operator[](int i) const noexcept
  {
    return m_data[i];
  }

  template <typename T>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  constexpr int
  Span<T>::length() const noexcept
  {
    return m_size;
  }

  template <typename T>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  constexpr const T*
  Span<T>::begin() const noexcept
  {
    return m_data;
  }

  template <typename T>
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  constexpr const T*
  Span<T>::end() const noexcept
  {
    return m_data + m_size;
  }
} // namespace EBGeometry

#endif
