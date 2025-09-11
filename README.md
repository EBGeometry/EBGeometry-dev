## EBGeometry

EBGeometry is a CPU/GPU code for

1. Turning watertight and orientable surface grids into signed distance functions (SDFs).
2. Fast evaluation of such grids using bounding volume hierarchies (BVHs).
3. Providing fast constructive solid geometry (CSG) unions using BVHs. 

This code is header-only and can be dropped into any C++ project that supports C++20.
It was originally written to be used with embedded-boundary (EB) codes like Chombo or AMReX.
However, EBGeometry provides quite general SDFs, implicit functions, and CSG unions, and is useful beyond those codes. 

To clone EBGeometry:

    git clone git@github.com:rmrsk/EBGeometry.git
	
## Requirements

* A C++ compiler which supports C++20.

EBGeometry is a header-only library in C++ and has no external dependencies.
To use it, simply make EBGeometry.hpp visible to your code and include it.	
	
## Building

To build EBGeometry, navigate to the top folder and perform the following

```
cmake -B build
cmake --build build
```

License
-------

See LICENSE and Copyright.txt for redistribution rights. 
