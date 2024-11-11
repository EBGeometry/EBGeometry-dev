#include "EBGeometry.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using namespace EBGeometry;

constexpr Real one   = 1.0;
constexpr Real two   = 2.0;
constexpr Real three = 3.0;

class OneIF : public ImplicitFunction
{
public:
  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] Real
  value(const Vec3& a_point) const noexcept override final
  {
    return one;
  }
};

class TwoIF : public ImplicitFunction
{
public:
  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] Real
  value(const Vec3& a_point) const noexcept override final
  {
    return two;
  }
};

class ThreeIF : public ImplicitFunction
{
public:
  EBGEOMETRY_GPU_HOST_DEVICE
  [[nodiscard]] Real
  value(const Vec3& a_point) const noexcept override final
  {
    return three;
  }
};

TEST_CASE("VirtualFunctions_Host_Construct_Free")
{
  auto oneIF   = createImpFunc<OneIF, MemoryLocation::Host>();
  auto twoIF   = createImpFunc<TwoIF, MemoryLocation::Host>();
  auto threeIF = createImpFunc<ThreeIF, MemoryLocation::Host>();

  CHECK(oneIF != nullptr);
  CHECK(twoIF != nullptr);
  CHECK(threeIF != nullptr);

  freeImpFunc(oneIF);
  freeImpFunc(twoIF);
  freeImpFunc(threeIF);
}

TEST_CASE("VirtualFunctions_Host_Inheritance")
{
  auto oneIF   = static_cast<ImplicitFunction*>(createImpFunc<OneIF, MemoryLocation::Host>());
  auto twoIF   = static_cast<ImplicitFunction*>(createImpFunc<TwoIF, MemoryLocation::Host>());
  auto threeIF = static_cast<ImplicitFunction*>(createImpFunc<ThreeIF, MemoryLocation::Host>());

  CHECK(oneIF->value(Vec3::zero()) == one);
  CHECK(twoIF->value(Vec3::zero()) == two);
  CHECK(threeIF->value(Vec3::zero()) == three);

  freeImpFunc(oneIF);
  freeImpFunc(twoIF);
  freeImpFunc(threeIF);
}

#ifdef EBGEOMETRY_ENABLE_GPU
TEST_CASE("VirtualFunctions_GPU_Inheritance")
{
  auto oneIF   = static_cast<ImplicitFunction*>(createImpFunc<OneIF, MemoryLocation::Global>());
  auto twoIF   = static_cast<ImplicitFunction*>(createImpFunc<TwoIF, MemoryLocation::Global>());
  auto threeIF = static_cast<ImplicitFunction*>(createImpFunc<ThreeIF, MemoryLocation::Global>());

  // CHECK(oneIF->value(Vec3::zero()) == one);
  // CHECK(twoIF->value(Vec3::zero()) == two);
  // CHECK(threeIF->value(Vec3::zero()) == three);

  freeImpFunc(oneIF);
  freeImpFunc(twoIF);
  freeImpFunc(threeIF);
}
#endif
