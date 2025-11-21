// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "EBGeometry.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using namespace EBGeometry;

EBGEOMETRY_GPU_GLOBAL
void
evalImpFunc(Real* value, ImplicitFunction* func, Vec3* point)
{
  *value = func->value(*point);
}

TEST_CASE("PlaneSDF_Construct_Free_Host")
{
  auto plane1_host = createImpFunc<PlaneSDF, MemoryLocation::Host>();
  auto plane2_host = createImpFunc<PlaneSDF, MemoryLocation::Host>(Vec3::one(), Vec3::one());

  CHECK(plane1_host != nullptr);
  CHECK(plane2_host != nullptr);

  freeImpFunc(plane1_host);
  freeImpFunc(plane2_host);
}

#ifdef EBGEOMETRY_ENABLE_CUDA
TEST_CASE("PlaneSDF_Construct_Free_CUDA")
{
  auto plane1_gpu = createImpFunc<PlaneSDF, MemoryLocation::Global>();
  auto plane2_gpu = createImpFunc<PlaneSDF, MemoryLocation::Global>(Vec3::one(), Vec3::one());
  auto plane3_gpu = createImpFunc<PlaneSDF, MemoryLocation::Unified>();
  auto plane4_gpu = createImpFunc<PlaneSDF, MemoryLocation::Unified>(Vec3::one(), Vec3::one());

  CHECK(plane1_gpu != nullptr);
  CHECK(plane2_gpu != nullptr);
  CHECK(plane3_gpu != nullptr);
  CHECK(plane4_gpu != nullptr);

  freeImpFunc(plane1_gpu);
  freeImpFunc(plane2_gpu);
  freeImpFunc(plane3_gpu);
  freeImpFunc(plane4_gpu);
}
#endif

TEST_CASE("PlaneSDF_Value_Host")
{
  auto plane1_host = createImpFunc<PlaneSDF, MemoryLocation::Host>(Vec3::zero(), +Vec3::one());
  auto plane2_host = createImpFunc<PlaneSDF, MemoryLocation::Host>(Vec3::one(), -Vec3::one());

  CHECK(plane1_host->value(+Vec3::one()) == +Real(sqrt(3.0)));
  CHECK(plane1_host->value(-Vec3::one()) == -Real(sqrt(3.0)));
  CHECK(plane1_host->value(Vec3::unit(2)) == +Real(1.0 / sqrt(3.0)));
  CHECK(plane1_host->value(-Vec3::unit(1)) == -Real(1.0 / sqrt(3.0)));

  CHECK(plane2_host->value(+Vec3::one()) == Real(0.0));
  CHECK(plane2_host->value(-Vec3::one()) == Real(sqrt(12.0)));

  freeImpFunc(plane1_host);
  freeImpFunc(plane2_host);
}
