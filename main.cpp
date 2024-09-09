#include "EBGeometry.hpp"

using namespace EBGeometry;

EBGEOMETRY_GPU_GLOBAL
void addNumbers(Vec3 a, Vec3 b) {
  const auto c = a + b;

  printf("hei");
}

EBGEOMETRY_GPU_HOST_DEVICE
std::pair<int,int> foo() {
  return {0,0};
};
  

int main()
{
  Vec2 v1;
  Vec2 v2;  
  Vec3 v3;  

  std::cout << v1 << std::endl;  
  std::cout << v2 << std::endl;
  std::cout << v3 << std::endl;

  addNumbers<<<1,1>>>(v3,v3);
  
  cudaDeviceSynchronize();
  
  return 0;
}
