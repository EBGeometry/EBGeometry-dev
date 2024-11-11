// Std includes
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Our includes
#include "EBGeometry.hpp"

//-------------------------------------------------
// These three classes are textbook polymorphism.
//-------------------------------------------------

class Base
{
public:
  __device__ virtual void
  doThing() = 0;
};

class Derived1 : public Base
{
  double firstvalue;
  double secondvalue;

public:
  __device__
  Derived1(double x, double y)
    : firstvalue(x), secondvalue(y)
  {}
  __device__ virtual void
  doThing() override
  {
    printf("Greetings! I am Derived1, and I contain the values\n"
           " %f and %f.\n",
           firstvalue,
           secondvalue);
  }
};

class Derived2 : public Base
{
  char firstchar;
  char secondchar;

public:
  __device__
  Derived2(char x, char y)
    : firstchar(x), secondchar(y)
  {}
  __device__ virtual void
  doThing() override
  {
    printf("Waddup! I am Derived2, with the two characters\n"
           " %c and %c.\n",
           firstchar,
           secondchar);
  }
};

//---------------------------------------------------------
// This kernel will (hopefully) call a polymorphic method.
//---------------------------------------------------------

__global__ void
call_virtual_method(Base* b)
{
  b->doThing();
}

__global__ void
call_der1_constructor(Derived1* d, double x, double y)
{
  new (d) Derived1(x, y);
}
__global__ void
call_der2_constructor(Derived2* d, char x, char y)
{
  new (d) Derived2(x, y);
}

int
main()
{

  // These will be pointers to objects on the device.
  Derived1* der1;
  Derived2* der2;

  cudaMalloc(&der1, sizeof(Derived1));
  cudaMalloc(&der2, sizeof(Derived2));

  call_der1_constructor<<<1, 1>>>(der1, 3.0, 5.0);
  call_der2_constructor<<<1, 1>>>(der2, 'c', 'g');

  Base* b1 = der1;
  Base* b2 = der2;

  call_virtual_method<<<1, 1>>>(b1);
  call_virtual_method<<<1, 1>>>(b2);

  cudaFree(der1);
  cudaFree(der2);
}
