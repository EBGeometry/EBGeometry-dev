/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_VecImplem.hpp
  @brief  Implementation of EBGeometry_Vec.hpp
  @author Robert Marskar
*/

#ifndef EBGeometry_VecImplem
#define EBGeometry_VecImplem

// Std includes
#include <math.h>
#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>

// Our includes
#include "EBGeometry_Vec.hpp"

namespace EBGeometry {

  inline Vec2::Vec2() noexcept
  {
    *this = Vec2::zero();
  }

  inline Vec2::Vec2(const Vec2& u) noexcept
  {
    this->m_x = u.m_x;
    this->m_y = u.m_y;
  }

  inline Vec2::~Vec2() noexcept
  {}

  inline Vec2::Vec2(const Real& a_x, const Real& a_y) noexcept
  {
    this->m_x = a_x;
    this->m_y = a_y;
  }

  inline Real&
  Vec2::x() noexcept
  {
    return (this->m_x);
  }

  inline const Real&
  Vec2::x() const noexcept
  {
    return (this->m_x);
  }

  inline Real&
  Vec2::y() noexcept
  {
    return (this->m_y);
  }

  inline const Real&
  Vec2::y() const noexcept
  {
    return (this->m_y);
  }

  inline Vec2
  Vec2::zero() noexcept
  {
    return Vec2(Real(0.0), Real(0.0));
  }

  inline Vec2
  Vec2::one() noexcept
  {
    return Vec2(Real(1.0), Real(1.0));
  }

  inline Vec2
  Vec2::min() noexcept
  {
    return Vec2(std::numeric_limits<Real>::lowest(), std::numeric_limits<Real>::lowest());
  }

  inline Vec2
  Vec2::max() noexcept
  {
    return Vec2(std::numeric_limits<Real>::max(), std::numeric_limits<Real>::max());
  }

  inline Vec2
  Vec2::infinity() noexcept
  {
    return Vec2(std::numeric_limits<Real>::infinity(), std::numeric_limits<Real>::infinity());
  }

  inline Vec2&
  Vec2::operator=(const Vec2& u) noexcept
  {
    this->m_x = u.m_x;
    this->m_y = u.m_y;

    return (*this);
  }

  inline Vec2
  Vec2::operator+(const Vec2& u) const noexcept
  {
    return Vec2(m_x + u.m_x, m_y + u.m_y);
  }

  inline Vec2
  Vec2::operator-(const Vec2& u) const noexcept
  {
    return Vec2(m_x - u.m_x, m_y - u.m_y);
  }

  inline Vec2
  Vec2::operator-() const noexcept
  {
    return Vec2(-m_x, -m_y);
  }

  inline Vec2
  Vec2::operator*(const Real& s) const noexcept
  {
    return Vec2(m_x * s, m_y * s);
  }

  inline Vec2
  Vec2::operator/(const Real& s) const noexcept
  {
    const Real is = 1. / s;

    return Vec2(m_x * is, m_y * is);
  }

  inline Vec2&
  Vec2::operator+=(const Vec2& u) noexcept
  {
    m_x += u.m_x;
    m_y += u.m_y;

    return (*this);
  }

  inline Vec2&
  Vec2::operator-=(const Vec2& u) noexcept
  {
    m_x -= u.m_x;
    m_y -= u.m_y;

    return (*this);
  }

  inline Vec2&
  Vec2::operator*=(const Real& s) noexcept
  {
    m_x *= s;
    m_y *= s;

    return (*this);
  }

  inline Vec2&
  Vec2::operator/=(const Real& s) noexcept
  {
    const Real is = 1. / s;

    m_x *= is;
    m_y *= is;

    return (*this);
  }

  inline Real
  Vec2::dot(const Vec2& u) const noexcept
  {
    return m_x * u.m_x + m_y * u.m_y;
  }

  inline Real
  Vec2::length() const noexcept
  {
    return sqrt(m_x * m_x + m_y * m_y);
  }

  inline Real
  Vec2::length2() const noexcept
  {
    return m_x * m_x + m_y * m_y;
  }

  inline Vec2
  operator*(const Real& s, const Vec2& a_other) noexcept
  {
    return a_other * s;
  }

  inline Vec2
  operator/(const Real& s, const Vec2& a_other) noexcept
  {
    return a_other / s;
  }

  inline Vec3::Vec3() noexcept
  {
    (*this) = Vec3::zero();
  }

  inline Vec3::Vec3(const Vec3& u) noexcept
  {
    m_X[0] = u[0];
    m_X[1] = u[1];
    m_X[2] = u[2];
  }

  inline Vec3::Vec3(const Real& a_x, const Real& a_y, const Real& a_z) noexcept
  {
    m_X[0] = a_x;
    m_X[1] = a_y;
    m_X[2] = a_z;
  }

  inline Vec3::~Vec3() noexcept
  {}

  inline Vec3
  Vec3::zero() noexcept
  {
    return Vec3(Real(0.0), Real(0.0), Real(0.0));
  }

  inline Vec3
  Vec3::one() noexcept
  {
    return Vec3(Real(1.0), Real(1.0), Real(1.0));
  }

  inline Vec3
  Vec3::unit(const size_t a_dir) noexcept
  {
    Vec3 v = Vec3::zero();

    v[a_dir] = 1.0;

    return v;
  }

  inline Vec3
  Vec3::min() noexcept
  {
    return Vec3(
      std::numeric_limits<Real>::lowest(), std::numeric_limits<Real>::lowest(), std::numeric_limits<Real>::lowest());
  }

  inline Vec3
  Vec3::max() noexcept
  {
    return Vec3(std::numeric_limits<Real>::max(), std::numeric_limits<Real>::max(), std::numeric_limits<Real>::max());
  }

  inline Vec3
  Vec3::infinity() noexcept
  {
    return Vec3(std::numeric_limits<Real>::infinity(),
                std::numeric_limits<Real>::infinity(),
                std::numeric_limits<Real>::infinity());
  }

  inline Vec3&
  Vec3::operator=(const Vec3& u) noexcept
  {
    m_X[0] = u[0];
    m_X[1] = u[1];
    m_X[2] = u[2];

    return (*this);
  }

  inline Vec3
  Vec3::operator+(const Vec3& u) const noexcept
  {
    return Vec3(m_X[0] + u[0], m_X[1] + u[1], m_X[2] + u[2]);
  }

  inline Vec3
  Vec3::operator-(const Vec3& u) const noexcept
  {
    return Vec3(m_X[0] - u[0], m_X[1] - u[1], m_X[2] - u[2]);
  }

  inline Vec3
  Vec3::operator-() const noexcept
  {
    return Vec3(-m_X[0], -m_X[1], -m_X[2]);
  }

  inline Vec3
  Vec3::operator*(const Real& s) const noexcept
  {
    return Vec3(s * m_X[0], s * m_X[1], s * m_X[2]);
  }

  inline Vec3
  Vec3::operator*(const Vec3& s) const noexcept
  {
    return Vec3(s[0] * m_X[0], s[1] * m_X[1], s[2] * m_X[2]);
  }

  inline Vec3
  Vec3::operator/(const Real& s) const noexcept
  {
    const Real is = 1. / s;
    return Vec3(is * m_X[0], is * m_X[1], is * m_X[2]);
  }

  inline Vec3
  Vec3::operator/(const Vec3& v) const noexcept
  {
    return Vec3(m_X[0] / v[0], m_X[1] / v[1], m_X[2] / v[2]);
  }

  inline Vec3&
  Vec3::operator+=(const Vec3& u) noexcept
  {
    m_X[0] += u[0];
    m_X[1] += u[1];
    m_X[2] += u[2];

    return (*this);
  }

  inline Vec3&
  Vec3::operator-=(const Vec3& u) noexcept
  {
    m_X[0] -= u[0];
    m_X[1] -= u[1];
    m_X[2] -= u[2];

    return (*this);
  }

  inline Vec3&
  Vec3::operator*=(const Real& s) noexcept
  {
    m_X[0] *= s;
    m_X[1] *= s;
    m_X[2] *= s;

    return (*this);
  }

  inline Vec3&
  Vec3::operator/=(const Real& s) noexcept
  {
    const Real is = 1. / s;

    m_X[0] *= is;
    m_X[1] *= is;
    m_X[2] *= is;

    return (*this);
  }

  inline Vec3
  Vec3::cross(const Vec3& u) const noexcept
  {
    return Vec3(m_X[1] * u[2] - m_X[2] * u[1], m_X[2] * u[0] - m_X[0] * u[2], m_X[0] * u[1] - m_X[1] * u[0]);
  }

  inline Real&
  Vec3::operator[](size_t i) noexcept
  {
    return m_X[i];
  }

  inline const Real&
  Vec3::operator[](size_t i) const noexcept
  {
    return m_X[i];
  }

  inline Vec3
  Vec3::min(const Vec3& u) noexcept
  {
    m_X[0] = std::min(m_X[0], u[0]);
    m_X[1] = std::min(m_X[1], u[1]);
    m_X[2] = std::min(m_X[2], u[2]);

    return *this;
  }

  inline Vec3
  Vec3::max(const Vec3& u) noexcept
  {
    m_X[0] = std::max(m_X[0], u[0]);
    m_X[1] = std::max(m_X[1], u[1]);
    m_X[2] = std::max(m_X[2], u[2]);

    return *this;
  }

  inline size_t
  Vec3::minDir(const bool a_doAbs) const noexcept
  {
    size_t mDir = 0;

    for (size_t dir = 0; dir < 3; dir++) {
      if (a_doAbs) {
        if (std::abs(m_X[dir]) < std::abs(m_X[mDir])) {
          mDir = dir;
        }
      }
      else {
        if (m_X[dir] < m_X[mDir]) {
          mDir = dir;
        }
      }
    }

    return mDir;
  }

  inline size_t
  Vec3::maxDir(const bool a_doAbs) const noexcept
  {
    size_t mDir = 0;

    for (size_t dir = 0; dir < 3; dir++) {
      if (a_doAbs) {
        if (std::abs(m_X[dir]) > std::abs(m_X[mDir])) {
          mDir = dir;
        }
      }
      else {
        if (m_X[dir] > m_X[mDir]) {
          mDir = dir;
        }
      }
    }

    return mDir;
  }

  inline bool
  Vec3::operator==(const Vec3& u) const noexcept
  {
    return (m_X[0] == u[0] && m_X[1] == u[1] && m_X[2] == u[2]);
  }

  inline bool
  Vec3::operator!=(const Vec3& u) const noexcept
  {
    return !(*this == u);
  }

  inline bool
  Vec3::operator<(const Vec3& u) const noexcept
  {
    return (m_X[0] < u[0] && m_X[1] < u[1] && m_X[2] < u[2]);
  }

  inline bool
  Vec3::operator>(const Vec3& u) const noexcept
  {
    return (m_X[0] > u[0] && m_X[1] > u[1] && m_X[2] > u[2]);
  }

  inline bool
  Vec3::operator<=(const Vec3& u) const noexcept
  {
    return (m_X[0] <= u[0] && m_X[1] <= u[1] && m_X[2] <= u[2]);
  }

  inline bool
  Vec3::operator>=(const Vec3& u) const noexcept
  {
    return (m_X[0] >= u[0] && m_X[1] >= u[1] && m_X[2] >= u[2]);
  }

  inline Real
  Vec3::dot(const Vec3& u) const noexcept
  {
    return m_X[0] * u[0] + m_X[1] * u[1] + m_X[2] * u[2];
  }

  inline Real
  Vec3::length() const noexcept
  {
    return sqrt(m_X[0] * m_X[0] + m_X[1] * m_X[1] + m_X[2] * m_X[2]);
  }

  inline Real
  Vec3::length2() const noexcept
  {
    return m_X[0] * m_X[0] + m_X[1] * m_X[1] + m_X[2] * m_X[2];
  }

  inline Vec2
  min(const Vec2& u, const Vec2& v) noexcept
  {
    return Vec2(std::min(u.x(), v.x()), std::min(u.y(), v.y()));
  }

  inline Vec2
  max(const Vec2& u, const Vec2& v) noexcept
  {
    return Vec2(std::max(u.x(), v.x()), std::max(u.y(), v.y()));
  }

  inline Real
  dot(const Vec2& u, const Vec2& v) noexcept
  {
    return u.dot(v);
  }

  inline Real
  length(const Vec2& v) noexcept
  {
    return v.length();
  }

  inline Vec3
  operator*(const Real& s, const Vec3& a_other) noexcept
  {
    return a_other * s;
  }

  inline Vec3
  operator/(const Real& s, const Vec3& a_other) noexcept
  {
    return a_other / s;
  }

  inline Vec3
  min(const Vec3& u, const Vec3& v) noexcept
  {
    return Vec3(std::min(u[0], v[0]), std::min(u[1], v[1]), std::min(u[2], v[2]));
  }

  inline Vec3
  max(const Vec3& u, const Vec3& v) noexcept
  {
    return Vec3(std::max(u[0], v[0]), std::max(u[1], v[1]), std::max(u[2], v[2]));
  }

  inline Real
  dot(const Vec3& u, const Vec3& v) noexcept
  {
    return u.dot(v);
  }

  inline Real
  length(const Vec3& v) noexcept
  {
    return v.length();
  }

} // namespace EBGeometry

#endif
