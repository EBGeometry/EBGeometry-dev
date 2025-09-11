#include "EBGeometry.hpp"

using namespace EBGeometry;

int
main()
{

  auto sphereHost = createImpFunc<SphereSDF, MemoryLocation::Host>(Vec3::zero(), Real(0.5));

  std::cout << "Evaluation host: value = " << sphereHost->value(Vec3::one()) << "\n";

  freeImpFunc(sphereHost);

#ifdef EBGEOMETRY_ENABLE_GPU
  auto sphereDevice = createImpFunc<SphereSDF, MemoryLocation::Pinned>(Vec3::zero(), Real(1.0));
#endif

  return 0;
}


