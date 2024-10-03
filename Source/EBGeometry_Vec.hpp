/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_Vec.hpp
  @brief  Declaration of 2D and 3D point/vector classes with templated
  precision. 
  @author Robert Marskar
*/

#ifndef EBGeometry_Vec
#define EBGeometry_Vec

// Std includes
#include <array>
#include <iostream>

// Our includes
#include "EBGeometry_GPU.hpp"
#include "EBGeometry_Types.hpp"

namespace EBGeometry {

  /*!
    @brief Two-dimensional vector class with arithmetic operators.
    @details The class has a public-only interface. To change it's components one
    can call the member functions, or set components directly, e.g. vec.x = 5.0
    @note Vec2 is primarily a utility class used together with DCEL signed distance
    functionality. 
  */
  class Vec2
  {
  public:
    /*!
      @brief For outputting a vector to an output stream. 
    */
    friend std::ostream&
    operator<<(std::ostream& os, const Vec2& vec)
    {
      os << '(' << vec.m_x << ',' << vec.m_y << ')';

      return os;
    }

    /*!
      @brief Default constructor. Sets the vector to the zero vector.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline Vec2() noexcept;

    /*!
      @brief Copy constructor
      @param[in] u Other vector
      @details Sets *this = u
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline Vec2(const Vec2& u) noexcept;

    /*!
      @brief Full constructor
      @param[in] a_x First vector component
      @param[in] a_y Second vector component
      @details Sets this->x = a_x and this->y = a_y
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline Vec2(const Real& a_x, const Real& a_y) noexcept;

    /*!
      @brief Destructor (does nothing)
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline ~Vec2() noexcept;

    /*!
      @brief Get x-component
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Real&
    x() noexcept;
    /*!
      @brief Get x-component
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline const Real&
    x() const noexcept;

    /*!
      @brief Get y-component
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Real&
    y() noexcept;

    /*!
      @brief Get y-component
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline const Real&
    y() const noexcept;

    /*!
      @brief Return av vector with x = y = 0
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline static Vec2
    zero() noexcept;

    /*!
      @brief Return av vector with x = y = 1
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline static Vec2
    one() noexcept;

    /*!
      @brief Return minimum possible representative vector.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline static Vec2
    min() noexcept;

    /*!
      @brief Return maximum possible representative vector.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline static Vec2
    max() noexcept;

    /*!
      @brief Assignment operator. Sets this.x = a_other.x and this.y = a_other.y
      @param[in] a_other Other vector
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline Vec2&
    operator=(const Vec2& a_other) noexcept;

    /*!
      @brief Addition operator.
      @param[in] a_other Other vector
      @details Returns a new object with component x = this->x + a_other.x (same
      for y-component)
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Vec2
    operator+(const Vec2& a_other) const noexcept;

    /*!
      @brief Subtraction operator.
      @param[in] a_other Other vector
      @details Returns a new object with component x = this->x - a_other.x (same
      for y-component)
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Vec2
    operator-(const Vec2& a_other) const noexcept;

    /*!
      @brief Negation operator. Returns a new Vec2 with negated components
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Vec2
    operator-() const noexcept;

    /*!
      @brief Multiplication operator
      @param[in] s Scalar to be multiplied
      @details Returns a new Vec2 with components x = s*this->x (and same for
      y)
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Vec2
    operator*(const Real& s) const noexcept;

    /*!
      @brief Division operator
      @param[in] s Scalar to be multiplied
      @details Returns a new Vec2 with components x = (1/s)*this->x (and same
      for y)
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Vec2
    operator/(const Real& s) const noexcept;

    /*!
      @brief Addition operator
      @param[in] a_other Other vector to add
      @details Returns (*this) with components this->x = this->x + a_other.x (and
      same for y)
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline Vec2&
    operator+=(const Vec2& a_other) noexcept;

    /*!
      @brief Subtraction operator
      @param[in] a_other Other vector to subtract
      @details Returns (*this) with components this->x = this->x - a_other.x (and
      same for y)
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline Vec2&
    operator-=(const Vec2& a_other) noexcept;

    /*!
      @brief Multiplication operator
      @param[in] s Scalar to multiply by
      @details Returns (*this) with components this->x = s*this->x (and same for
      y)
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline Vec2&
    operator*=(const Real& s) noexcept;

    /*!
      @brief Division operator operator
      @param[in] s Scalar to divide by
      @details Returns (*this) with components this->x = (1/s)*this->x (and same
      for y)
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline Vec2&
    operator/=(const Real& s) noexcept;

    /*!
      @brief Dot product operator
      @param[in] a_other other vector
      @details Returns the dot product, i.e. this->x*a_other.x + this->y+a_other.y
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Real
    dot(const Vec2& a_other) const noexcept;

    /*!
      @brief Compute length of vector
      @return Returns length of vector, i.e. sqrt[(this->x)*(this->x) +
      (this->y)*(this->y)]
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Real
    length() const noexcept;

    /*!
      @brief Compute square of vector
      @return Returns length of vector, i.e. (this->x)*(this->x) +
      (this->y)*(this->y)
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Real
    length2() const noexcept;

  protected:
    /*!
      @brief First component in the vector
    */
    Real m_x;

    /*!
      @brief Second component in the vector
    */
    Real m_y;
  };

  /*!
    @brief Three-dimensional vector class with arithmetic operators.
    @details The class has a public-only interface. To change it's components one
    can call the member functions, or set components directly, e.g. vec.x = 5.0
  */
  class Vec3
  {
  public:
    /*!
      @brief For outputting a vector to an output stream. 
    */
    friend std::ostream&
    operator<<(std::ostream& os, const Vec3& vec)
    {
      os << '(' << vec[0] << ',' << vec[1] << ',' << vec[2] << ')';

      return os;
    }

    /*!
      @brief Default constructor. Sets the vector to the zero vector.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline Vec3() noexcept;

    /*!
      @brief Copy constructor
      @param[in] a_u Other vector
      @details Sets *this = u
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline Vec3(const Vec3& a_u) noexcept;

    /*!
      @brief Full constructor
      @param[in] a_x First vector component
      @param[in] a_y Second vector component
      @param[in] a_z Third vector component
      @details Sets this->x = a_x, this->y = a_y, and this->z = a_z
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline Vec3(const Real& a_x, const Real& a_y, const Real& a_z) noexcept;

    /*!
      @brief Destructor (does nothing)
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline ~Vec3() noexcept;

    /*!
      @brief Return av vector with x = y = z = 0
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline static Vec3
    zero() noexcept;

    /*!
      @brief Return av vector with x = y = z = 1
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline static Vec3
    one() noexcept;

    /*!
      @brief Return av vector with x = y = z = 1
      @param[in] a_dir Dircetion
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline static Vec3
    unit(const size_t a_dir) noexcept;

    /*!
      @brief Return a vector with minimum representable components.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline static Vec3
    min() noexcept;

    /*!
      @brief Return a vector with maximum representable components.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline static Vec3
    max() noexcept;

    /*!
      @brief Return a vector with maximally negative representable components.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline static Vec3
    lowest() noexcept;

    /*!
      @brief Return component in vector. (i=0 => x and so on)
      @param[in] i Index. Must be < 3
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Real&
    operator[](size_t i) noexcept;

    /*!
      @brief Return non-modifiable component in vector. (i=0 => x and so on)
      @param[in] i Index. Must be < 3
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline const Real&
    operator[](size_t i) const noexcept;

    /*!
      @brief Comparison operator. Returns true if all components are the same
      @param[in] u Other vector
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline bool
    operator==(const Vec3& u) const noexcept;

    /*!
      @brief Comparison operator. Returns false if all components are the same
      @param[in] u Other vector
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline bool
    operator!=(const Vec3& u) const noexcept;

    /*!
      @brief "Smaller than" operator.
      @details Returns true if this->x < u.x AND this->y < u.y AND this->z < u.z
      and false otherwise
      @param[in] u Other vector
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline bool
    operator<(const Vec3& u) const noexcept;

    /*!
      @brief "Greater than" operator.
      @details Returns true if this->x > u.x AND this->y > u.y AND this->z > u.z
      @param[in] u Other vector
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline bool
    operator>(const Vec3& u) const noexcept;

    /*!
      @brief "Smaller or equal to" operator.
      @details Returns true if this->x <= u.x AND this->y <= u.y AND this->z <=
      u.z
      @param[in] u Other vector
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline bool
    operator<=(const Vec3& u) const noexcept;

    /*!
      @brief Lexicographical comparison operator
      @details Returns true if this Vec3 is lexicographically smaller than the other
      @param[in] u Other vector
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline bool
    lessLX(const Vec3& u) const noexcept;

    /*!
      @brief "Greater or equal to" operator.
      @details Returns true if this->x >= u.x AND this->y >= u.y AND this->z >=
      u.z
      @param[in] u Other vector
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline bool
    operator>=(const Vec3& u) const noexcept;

    /*!
      @brief Assignment operator. Sets components equal to the argument vector's
      components
      @param[in] u Other vector
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline Vec3&
    operator=(const Vec3& u) noexcept;

    /*!
      @brief Addition operator. Returns a new vector with added components
      @return Returns a new vector with x = this->x - u.x and so on.
      @param[in] u Other vector
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Vec3
    operator+(const Vec3& u) const noexcept;

    /*!
      @brief Subtraction operator. Returns a new vector with subtracted components
      @return Returns a new vector with x = this->x - u.x and so on.
      @param[in] u Other vector
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Vec3
    operator-(const Vec3& u) const noexcept;

    /*!
      @brief Negation operator. Returns a vector with negated components
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Vec3
    operator-() const noexcept;

    /*!
      @brief Multiplication operator. Returns a vector with scalar multiplied
      components
      @param[in] s Scalar to multiply by
      @return Returns a new vector with X[i] = this->X[i] * s
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Vec3
    operator*(const Real& s) const noexcept;

    /*!
      @brief Component-wise multiplication operator
      @param[in] s Scalar to multiply by
      @return Returns a new vector with X[i] = this->X[i] * s[i] for each
      component.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Vec3
    operator*(const Vec3& s) const noexcept;

    /*!
      @brief Division operator. Returns a vector with scalar divided components
      @param[in] s Scalar to divided by
      @return Returns a new vector with X[i] = this->X[i] / s
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Vec3
    operator/(const Real& s) const noexcept;

    /*!
      @brief Component-wise division operator.
      @param[in] v Other vector
      @return Returns a new vector with X[i] = this->X[i]/v[i] for each component.
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Vec3
    operator/(const Vec3& v) const noexcept;

    /*!
      @brief Vector addition operator.
      @param[in] u Vector to add
      @return Returns (*this) with incremented components, e.g. this->X[0] =
      this->X[0] + u.X[0]
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline Vec3&
    operator+=(const Vec3& u) noexcept;

    /*!
      @brief Vector subtraction operator.
      @param[in] u Vector to subtraction
      @return Returns (*this) with subtracted components, e.g. this->X[0] =
      this->X[0] - u.X[0]
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline Vec3&
    operator-=(const Vec3& u) noexcept;

    /*!
      @brief Vector multiplication operator.
      @param[in] s Scalar to multiply by
      @return Returns (*this) with multiplied components, e.g. this->X[0] =
      this->X[0] * s
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline Vec3&
    operator*=(const Real& s) noexcept;

    /*!
      @brief Vector division operator.
      @param[in] s Scalar to divide by
      @return Returns (*this) with multiplied components, e.g. this->X[0] =
      this->X[0] / s
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    inline Vec3&
    operator/=(const Real& s) noexcept;

    /*!
      @brief Vector minimum function. Returns a new vector with componentwise
      minimums.
      @param[in] u Other vector
      @return Returns a new vector with X[0] = std::min(this->X[0], u.X[0]) (and
      same for the other components)
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Vec3
    min(const Vec3& u) noexcept;

    /*!
      @brief Vector maximum function. Returns a new vector with componentwise
      maximums.
      @param[in] u Other vector
      @return Returns a new vector with X[0] = std::minmax->X[0], u.X[0]) (and
      same for the other components)
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Vec3
    max(const Vec3& u) noexcept;

    /*!
      @brief Vector cross product
      @param[in] u Other vector
      @returns Returns the cross product between (*this) and u
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Vec3
    cross(const Vec3& u) const noexcept;

    /*!
      @brief Vector dot product
      @param[in] u Other vector
      @returns Returns the dot product between (*this) and u
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Real
    dot(const Vec3& u) const noexcept;

    /*!
      @brief Return the direction which has the smallest component (can be
      absolute)
      @param[in] a_doAbs If true, evaluate component magnitudes rather than
      values.
      @return Direction with the biggest component
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline size_t
    minDir(const bool a_doAbs) const noexcept;

    /*!
      @brief Return the direction which has the largest component (can be
      absolute)
      @param[in] a_doAbs If true, evaluate component magnitudes rather than
      values.
      @return Direction with the biggest component
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline size_t
    maxDir(const bool a_doAbs) const noexcept;

    /*!
      @brief Compute vector length
      @return Returns the vector length, i.e. sqrt(X[0]*X[0] + X[1]*X[1] +
      Y[0]*Y[0])
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Real
    length() const noexcept;

    /*!
      @brief Compute vector length squared
      @return Returns the vector length squared, i.e. (X[0]*X[0] + X[1]*X[1] +
      Y[0]*Y[0])
    */
    EBGEOMETRY_GPU_HOST_DEVICE
    [[nodiscard]] inline Real
    length2() const noexcept;

  protected:
    /*!
      @brief Vector components
    */
    Real m_X[3];
  };

  /*!
    @brief Multiplication operator in the form s*Vec2
    @param[in] s Multiplication factor
    @param[in] a_other Other vector
    @return Returns a new vector with components x = s*a_other.x (and same for y)
  */
  EBGEOMETRY_GPU_HOST_DEVICE
  inline Vec2
  operator*(const Real& s, const Vec2& a_other) noexcept;

  /*!
    @brief Division operator in the form s*Vec2.
    @param[in] s Division factor
    @param[in] a_other Other vector
    @return Returns a new vector with components x = (1/s)*a_other.x (and same for
    y)
  */
  EBGEOMETRY_GPU_HOST_DEVICE
  inline Vec2
  operator/(const Real& s, const Vec2& a_other) noexcept;

  /*!
    @brief Minimum function. Returns new vector with component-wise minimums.
    @param[in] u Vector
    @param[in] v Other vector
    @return Returns new vector with components x = std::min(u.x, v.x).
  */
  EBGEOMETRY_GPU_HOST_DEVICE
  inline Vec2
  min(const Vec2& u, const Vec2& v) noexcept;

  /*!
    @brief Maximum function. Returns new vector with component-wise minimums.
    @param[in] u Vector
    @param[in] v Other vector
    @return Returns new vector with components x = std::max(u.x, v.x).
  */
  EBGEOMETRY_GPU_HOST_DEVICE
  inline Vec2
  max(const Vec2& u, const Vec2& v) noexcept;

  /*!
    @brief Dot product function.
    @param[in] u Vector
    @param[in] v Other vector
  */
  EBGEOMETRY_GPU_HOST_DEVICE
  inline Real
  dot(const Vec2& u, const Vec2& v) noexcept;

  /*!
    @brief Length function
    @param[in] v Vector.
  */
  EBGEOMETRY_GPU_HOST_DEVICE
  inline Real
  length(const Vec2& v) noexcept;

  /*!
    @brief Multiplication operator.
    @param[in] s Multiplication scalar
    @param[in] u Vector
    @return Returns new vector with components X[0] = s*X[0] and so on.
  */
  EBGEOMETRY_GPU_HOST_DEVICE
  inline Vec3
  operator*(const Real& s, const Vec3& u) noexcept;

  /*!
    @brief Division operator.
    @param[in] s Division scalar
    @param[in] u Vector
    @return Returns new vector with components X[0] = X[0]/s and so on.
  */
  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] inline Vec3
  operator/(const Real& s, const Vec3& u) noexcept;

  /*!
    @brief Minimum function. Returns new vector with component-wise minimums.
    @param[in] u Vector
    @param[in] v Other vector
    @return Returns new vector with components X[0] = std::min(u.X[0], v.X[0]) and
    so on
  */
  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] inline Vec3
  min(const Vec3& u, const Vec3& v) noexcept;

  /*!
    @brief Maximum function. Returns new vector with component-wise minimums.
    @param[in] u Vector
    @param[in] v Other vector
    @return Returns new vector with components X[0] = std::max(u.X[0], v.X[0]) and
    so on
  */
  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] inline Vec3
  max(const Vec3& u, const Vec3& v) noexcept;

  /*!
    @brief Dot product function.
    @param[in] u Vector
    @param[in] v Other vector
  */
  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] inline Real
  dot(const Vec3& u, const Vec3& v) noexcept;

  /*!
    @brief Length function
    @param[in] v Vector.
  */
  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] inline Real
  length(const Vec3& v) noexcept;

} // namespace EBGeometry

#include "EBGeometry_VecImplem.hpp"

#endif
