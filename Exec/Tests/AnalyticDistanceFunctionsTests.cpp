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

TEST_CASE("SphereSDF_Construct_Free")
{
  // Host construction.
  auto sphere_host = createImpFunc<SphereSDF, MemoryLocation::Host>();
  CHECK(sphere_host != nullptr);
  freeImpFunc(sphere_host);

  // CUDA construction.
#if EBGEOMETRY_ENABLE_CUDA
  auto sphere_gpu = createImpFunc<SphereSDF, MemoryLocation::Global>();
  CHECK(sphere_gpu != nullptr);
  freeImpFunc(sphere_gpu);
#endif  
}
