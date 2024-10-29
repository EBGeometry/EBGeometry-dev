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
#include <cmath>

// Our includes
#include "EBGeometry_GPUTypes.hpp"
#include "EBGeometry_Macros.hpp"
#include "EBGeometry_Vec.hpp"

namespace EBGeometry {

  EBGEOMETRY_ALWAYS_INLINE
  Vec2::Vec2() noexcept
  {
    *this = Vec2::zero();
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec2::Vec2(const Vec2& u) noexcept
  {
    this->m_x = u.m_x;
    this->m_y = u.m_y;
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec2::~Vec2() noexcept
  {}

  EBGEOMETRY_ALWAYS_INLINE
  Vec2::Vec2(const Real& a_x, const Real& a_y) noexcept
  {
    this->m_x = a_x;
    this->m_y = a_y;
  }

  EBGEOMETRY_ALWAYS_INLINE
  Real&
  Vec2::x() noexcept
  {
    return (this->m_x);
  }

  EBGEOMETRY_ALWAYS_INLINE
  const Real&
  Vec2::x() const noexcept
  {
    return (this->m_x);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Real&
  Vec2::y() noexcept
  {
    return (this->m_y);
  }

  EBGEOMETRY_ALWAYS_INLINE
  const Real&
  Vec2::y() const noexcept
  {
    return (this->m_y);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec2
  Vec2::zero() noexcept
  {
    return Vec2(Real(0.0), Real(0.0));
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec2
  Vec2::one() noexcept
  {
    return Vec2(Real(1.0), Real(1.0));
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec2
  Vec2::min() noexcept
  {
    return Vec2(EBGeometry::Limits::lowest(), EBGeometry::Limits::lowest());
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec2
  Vec2::max() noexcept
  {
    return Vec2(EBGeometry::Limits::max(), EBGeometry::Limits::max());
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec2&
  Vec2::operator=(const Vec2& u) noexcept
  {
    this->m_x = u.m_x;
    this->m_y = u.m_y;

    return (*this);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec2
  Vec2::operator+(const Vec2& u) const noexcept
  {
    return Vec2(m_x + u.m_x, m_y + u.m_y);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec2
  Vec2::operator-(const Vec2& u) const noexcept
  {
    return Vec2(m_x - u.m_x, m_y - u.m_y);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec2
  Vec2::operator-() const noexcept
  {
    return Vec2(-m_x, -m_y);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec2
  Vec2::operator*(const Real& s) const noexcept
  {
    return Vec2(m_x * s, m_y * s);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec2
  Vec2::operator/(const Real& s) const noexcept
  {
    const Real is = 1. / s;

    return Vec2(m_x * is, m_y * is);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec2&
  Vec2::operator+=(const Vec2& u) noexcept
  {
    m_x += u.m_x;
    m_y += u.m_y;

    return (*this);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec2&
  Vec2::operator-=(const Vec2& u) noexcept
  {
    m_x -= u.m_x;
    m_y -= u.m_y;

    return (*this);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec2&
  Vec2::operator*=(const Real& s) noexcept
  {
    m_x *= s;
    m_y *= s;

    return (*this);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec2&
  Vec2::operator/=(const Real& s) noexcept
  {
    const Real is = 1. / s;

    m_x *= is;
    m_y *= is;

    return (*this);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Real
  Vec2::dot(const Vec2& u) const noexcept
  {
    return m_x * u.m_x + m_y * u.m_y;
  }

  EBGEOMETRY_ALWAYS_INLINE
  Real
  Vec2::length() const noexcept
  {
    return sqrt(m_x * m_x + m_y * m_y);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Real
  Vec2::length2() const noexcept
  {
    return m_x * m_x + m_y * m_y;
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec2
  operator*(const Real& s, const Vec2& a_other) noexcept
  {
    return a_other * s;
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec2
  operator/(const Real& s, const Vec2& a_other) noexcept
  {
    return a_other / s;
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3::Vec3() noexcept
  {
    (*this) = Vec3::zero();
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3::Vec3(const Vec3& u) noexcept
  {
    m_X[0] = u[0];
    m_X[1] = u[1];
    m_X[2] = u[2];
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3::Vec3(const Real& a_x, const Real& a_y, const Real& a_z) noexcept
  {
    m_X[0] = a_x;
    m_X[1] = a_y;
    m_X[2] = a_z;
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3::~Vec3() noexcept
  {}

  EBGEOMETRY_ALWAYS_INLINE
  Vec3
  Vec3::zero() noexcept
  {
    return Vec3(Real(0.0), Real(0.0), Real(0.0));
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3
  Vec3::one() noexcept
  {
    return Vec3(Real(1.0), Real(1.0), Real(1.0));
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3
  Vec3::unit(const size_t a_dir) noexcept
  {
    Vec3 v = Vec3::zero();

    v[a_dir] = 1.0;

    return v;
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3
  Vec3::min() noexcept
  {
    return Vec3(EBGeometry::Limits::min(), EBGeometry::Limits::min(), EBGeometry::Limits::min());
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3
  Vec3::max() noexcept
  {
    return Vec3(EBGeometry::Limits::max(), EBGeometry::Limits::max(), EBGeometry::Limits::max());
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3
  Vec3::lowest() noexcept
  {
    return Vec3(EBGeometry::Limits::lowest(), EBGeometry::Limits::lowest(), EBGeometry::Limits::lowest());
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3&
  Vec3::operator=(const Vec3& u) noexcept
  {
    m_X[0] = u[0];
    m_X[1] = u[1];
    m_X[2] = u[2];

    return (*this);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3
  Vec3::operator+(const Vec3& u) const noexcept
  {
    return Vec3(m_X[0] + u[0], m_X[1] + u[1], m_X[2] + u[2]);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3
  Vec3::operator-(const Vec3& u) const noexcept
  {
    return Vec3(m_X[0] - u[0], m_X[1] - u[1], m_X[2] - u[2]);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3
  Vec3::operator-() const noexcept
  {
    return Vec3(-m_X[0], -m_X[1], -m_X[2]);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3
  Vec3::operator*(const Real& s) const noexcept
  {
    return Vec3(s * m_X[0], s * m_X[1], s * m_X[2]);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3
  Vec3::operator*(const Vec3& s) const noexcept
  {
    return Vec3(s[0] * m_X[0], s[1] * m_X[1], s[2] * m_X[2]);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3
  Vec3::operator/(const Real& s) const noexcept
  {
    const Real is = 1. / s;
    return Vec3(is * m_X[0], is * m_X[1], is * m_X[2]);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3
  Vec3::operator/(const Vec3& v) const noexcept
  {
    return Vec3(m_X[0] / v[0], m_X[1] / v[1], m_X[2] / v[2]);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3&
  Vec3::operator+=(const Vec3& u) noexcept
  {
    m_X[0] += u[0];
    m_X[1] += u[1];
    m_X[2] += u[2];

    return (*this);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3&
  Vec3::operator-=(const Vec3& u) noexcept
  {
    m_X[0] -= u[0];
    m_X[1] -= u[1];
    m_X[2] -= u[2];

    return (*this);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3&
  Vec3::operator*=(const Real& s) noexcept
  {
    m_X[0] *= s;
    m_X[1] *= s;
    m_X[2] *= s;

    return (*this);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3&
  Vec3::operator/=(const Real& s) noexcept
  {
    const Real is = 1. / s;

    m_X[0] *= is;
    m_X[1] *= is;
    m_X[2] *= is;

    return (*this);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3
  Vec3::cross(const Vec3& u) const noexcept
  {
    return Vec3(m_X[1] * u[2] - m_X[2] * u[1], m_X[2] * u[0] - m_X[0] * u[2], m_X[0] * u[1] - m_X[1] * u[0]);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Real&
  Vec3::operator[](size_t i) noexcept
  {
    EBGEOMETRY_EXPECT(i <= 2);

    return m_X[i];
  }

  EBGEOMETRY_ALWAYS_INLINE
  const Real&
  Vec3::operator[](size_t i) const noexcept
  {
    EBGEOMETRY_EXPECT(i <= 2);

    return m_X[i];
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3
  Vec3::min(const Vec3& u) noexcept
  {
    m_X[0] = EBGeometry::min(m_X[0], u[0]);
    m_X[1] = EBGeometry::min(m_X[1], u[1]);
    m_X[2] = EBGeometry::min(m_X[2], u[2]);

    return *this;
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3
  Vec3::max(const Vec3& u) noexcept
  {
    m_X[0] = EBGeometry::max(m_X[0], u[0]);
    m_X[1] = EBGeometry::max(m_X[1], u[1]);
    m_X[2] = EBGeometry::max(m_X[2], u[2]);

    return *this;
  }

  EBGEOMETRY_ALWAYS_INLINE
  size_t
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

  EBGEOMETRY_ALWAYS_INLINE
  size_t
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

  EBGEOMETRY_ALWAYS_INLINE
  bool
  Vec3::operator==(const Vec3& u) const noexcept
  {
    return (m_X[0] == u[0] && m_X[1] == u[1] && m_X[2] == u[2]);
  }

  EBGEOMETRY_ALWAYS_INLINE
  bool
  Vec3::operator!=(const Vec3& u) const noexcept
  {
    return !(*this == u);
  }

  EBGEOMETRY_ALWAYS_INLINE
  bool
  Vec3::operator<(const Vec3& u) const noexcept
  {
    return (m_X[0] < u[0] && m_X[1] < u[1] && m_X[2] < u[2]);
  }

  EBGEOMETRY_ALWAYS_INLINE
  bool
  Vec3::operator>(const Vec3& u) const noexcept
  {
    return (m_X[0] > u[0] && m_X[1] > u[1] && m_X[2] > u[2]);
  }

  EBGEOMETRY_ALWAYS_INLINE
  bool
  Vec3::operator<=(const Vec3& u) const noexcept
  {
    return (m_X[0] <= u[0] && m_X[1] <= u[1] && m_X[2] <= u[2]);
  }

  EBGEOMETRY_ALWAYS_INLINE
  bool
  Vec3::operator>=(const Vec3& u) const noexcept
  {
    return (m_X[0] >= u[0] && m_X[1] >= u[1] && m_X[2] >= u[2]);
  }

  EBGEOMETRY_ALWAYS_INLINE
  bool
  Vec3::lessLX(const Vec3& u) const noexcept
  {
    return (m_X[0] < u.m_X[0]) || (m_X[0] == u.m_X[0] && m_X[1] < u.m_X[1]) ||
           (m_X[1] == u.m_X[1] && m_X[2] < u.m_X[2]);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Real
  Vec3::dot(const Vec3& u) const noexcept
  {
    return m_X[0] * u[0] + m_X[1] * u[1] + m_X[2] * u[2];
  }

  EBGEOMETRY_ALWAYS_INLINE
  Real
  Vec3::length() const noexcept
  {
    return sqrt(m_X[0] * m_X[0] + m_X[1] * m_X[1] + m_X[2] * m_X[2]);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Real
  Vec3::length2() const noexcept
  {
    return m_X[0] * m_X[0] + m_X[1] * m_X[1] + m_X[2] * m_X[2];
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec2
  min(const Vec2& u, const Vec2& v) noexcept
  {
    return Vec2(EBGeometry::min(u.x(), v.x()), EBGeometry::min(u.y(), v.y()));
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec2
  max(const Vec2& u, const Vec2& v) noexcept
  {
    return Vec2(EBGeometry::max(u.x(), v.x()), EBGeometry::max(u.y(), v.y()));
  }

  EBGEOMETRY_ALWAYS_INLINE
  Real
  dot(const Vec2& u, const Vec2& v) noexcept
  {
    return u.dot(v);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Real
  length(const Vec2& v) noexcept
  {
    return v.length();
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3
  operator*(const Real& s, const Vec3& a_other) noexcept
  {
    return a_other * s;
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3
  operator/(const Real& s, const Vec3& a_other) noexcept
  {
    return a_other / s;
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3
  min(const Vec3& u, const Vec3& v) noexcept
  {
    return Vec3(EBGeometry::min(u[0], v[0]), EBGeometry::min(u[1], v[1]), EBGeometry::min(u[2], v[2]));
  }

  EBGEOMETRY_ALWAYS_INLINE
  Vec3
  max(const Vec3& u, const Vec3& v) noexcept
  {
    return Vec3(EBGeometry::max(u[0], v[0]), EBGeometry::max(u[1], v[1]), EBGeometry::max(u[2], v[2]));
  }

  EBGEOMETRY_ALWAYS_INLINE
  Real
  dot(const Vec3& u, const Vec3& v) noexcept
  {
    return u.dot(v);
  }

  EBGEOMETRY_ALWAYS_INLINE
  Real
  length(const Vec3& v) noexcept
  {
    return v.length();
  }

} // namespace EBGeometry

#endif
