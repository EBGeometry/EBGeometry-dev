// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "EBGeometry.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using namespace EBGeometry;

constexpr Real one   = 1.0;
constexpr Real two   = 2.0;
constexpr Real three = 3.0;

EBGEOMETRY_GPU_GLOBAL
void
evalImpFunc(Real* value, ImplicitFunction* func, Vec3* point)
{
  *value = func->value(*point);
}

class OneIF : public ImplicitFunction
{
public:
  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  Real
  value(const Vec3& a_point) const noexcept override
  {
    return one;
  }
};

class TwoIF : public OneIF
{
public:
  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  Real
  value(const Vec3& a_point) const noexcept override
  {
    return two;
  }
};

class SumIF : public ImplicitFunction
{
public:
  EBGEOMETRY_GPU_HOST_DEVICE
  EBGEOMETRY_ALWAYS_INLINE
  SumIF(const ImplicitFunction* const A, const ImplicitFunction* const B) noexcept :
    m_A(A),
    m_B(B)
  {
    EBGEOMETRY_ALWAYS_EXPECT(m_A != nullptr);
    EBGEOMETRY_ALWAYS_EXPECT(m_B != nullptr);
  }

  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
  Real
  value(const Vec3& a_point) const noexcept override final
  {
    return m_A->value(a_point) + m_B->value(a_point);
  }

protected:
  const ImplicitFunction* m_A;
  const ImplicitFunction* m_B;
};

TEST_CASE("VirtualFunctions_Host_Construct_Free")
{
  auto oneIF = createImpFunc<OneIF, MemoryLocation::Host>();
  auto twoIF = createImpFunc<TwoIF, MemoryLocation::Host>();
  auto sumIF = createImpFunc<SumIF, MemoryLocation::Host>(oneIF, twoIF);

  CHECK(oneIF != nullptr);
  CHECK(twoIF != nullptr);
  CHECK(sumIF != nullptr);

  freeImpFunc(oneIF);
  freeImpFunc(twoIF);
  freeImpFunc(sumIF);
}

TEST_CASE("VirtualFunctions_Host_Inheritance")
{
  auto oneIF = static_cast<ImplicitFunction*>(createImpFunc<OneIF, MemoryLocation::Host>());
  auto twoIF = static_cast<ImplicitFunction*>(createImpFunc<TwoIF, MemoryLocation::Host>());
  auto sumIF = static_cast<ImplicitFunction*>(createImpFunc<SumIF, MemoryLocation::Host>(oneIF, twoIF));

  CHECK(oneIF->value(Vec3::zero()) == one);
  CHECK(twoIF->value(Vec3::zero()) == two);
  CHECK(sumIF->value(Vec3::zero()) == (one + two));

  freeImpFunc(oneIF);
  freeImpFunc(twoIF);
  freeImpFunc(sumIF);
}

#ifdef EBGEOMETRY_ENABLE_CUDA
TEST_CASE("VirtualFunctions_CUDA_Inheritance")
{
  auto oneIF = static_cast<ImplicitFunction*>(createImpFunc<OneIF, MemoryLocation::Global>());
  auto twoIF = static_cast<ImplicitFunction*>(createImpFunc<TwoIF, MemoryLocation::Global>());
  auto sumIF = static_cast<ImplicitFunction*>(createImpFunc<SumIF, MemoryLocation::Global>(oneIF, twoIF));

  CHECK(oneIF != nullptr);
  CHECK(twoIF != nullptr);
  CHECK(sumIF != nullptr);

  Real  value_h;
  Real* value_d;
  Vec3* point_d;

  cudaMalloc((void**)&value_d, sizeof(Real));
  cudaMalloc((void**)&point_d, sizeof(Vec3));

  // OneIF
  evalImpFunc<<<1, 1>>>(value_d, oneIF, point_d);
  cudaMemcpy(&value_h, value_d, sizeof(Real), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  CHECK(value_h == one);

  // TwoIF
  evalImpFunc<<<1, 1>>>(value_d, twoIF, point_d);
  cudaMemcpy(&value_h, value_d, sizeof(Real), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  CHECK(value_h == two);

  // SumIF
  evalImpFunc<<<1, 1>>>(value_d, sumIF, point_d);
  cudaMemcpy(&value_h, value_d, sizeof(Real), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  CHECK(value_h == (one + two));

  freeImpFunc(oneIF);
  freeImpFunc(twoIF);
  freeImpFunc(sumIF);

  cudaFree(value_d);
  cudaFree(point_d);
}
#endif
