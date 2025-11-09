# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

EBGeometry is a header-only C++20 library for signed distance functions (SDFs) and constructive solid geometry (CSG). It converts watertight surface grids into SDFs and provides fast evaluation using bounding volume hierarchies (BVHs). The library supports both CPU and GPU (CUDA/HIP) execution.

## Build Commands

### Basic Build
```bash
cmake -B build
cmake --build build
```

### Build with Options
```bash
# Debug mode (enables assertions via EBGEOMETRY_ENABLE_DEBUG)
cmake -B build -DENABLE_DEBUG=ON

# Enable tests (requires Catch2)
cmake -B build -DENABLE_TESTS=ON
cmake --build build

# Enable examples
cmake -B build -DENABLE_EXAMPLES=ON

# GPU support (CUDA)
cmake -B build -DENABLE_CUDA=ON

# GPU support (HIP)
cmake -B build -DENABLE_HIP=ON

# Use double precision instead of float
cmake -B build -DENABLE_DOUBLE=ON

# Generate compile_commands.json for tooling
cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

### Running Tests
```bash
# Build tests first
cmake -B build -DENABLE_TESTS=ON
cmake --build build

# Run all tests via CTest
cd build && ctest

# Run individual test executables
./build/VecTests
./build/TriangleTests
./build/AnalyticDistanceFunctionsTests
```

### Running a Single Test
Tests use Catch2, so you can filter by test name:
```bash
./build/VecTests "[Vec3_Arithmetic]"
./build/TriangleTests "[Triangle_SignedDistance]"
```

### Code Quality Checks
```bash
# Run pre-commit hooks manually
pre-commit run --all-files

# Format code with clang-format
clang-format -i Source/*.hpp Exec/**/*.cpp

# Run clang-tidy (requires compile_commands.json)
clang-tidy -p build Exec/Examples/EBGeometry_Shapes/*.cpp
```

## Architecture

### Header-Only Design Pattern
- **Main entry point**: `EBGeometry.hpp` - include this single header to use the library
- **Pattern**: Declaration/Implementation split
  - Declaration files: `EBGeometry_*.hpp` (e.g., `EBGeometry_Vec.hpp`)
  - Implementation files: `EBGeometry_*Implem.hpp` (e.g., `EBGeometry_VecImplem.hpp`)
  - Implementation files are included at the end of declaration files
- **Source directory**: All library code lives in `Source/`

### Core Component Hierarchy

1. **Primitives** (`EBGeometry_Vec.hpp`, `EBGeometry_Triangle.hpp`)
   - `Vec2`, `Vec3`: SIMD-aligned vector types with full arithmetic operators
   - `Triangle<MetaData>`: Self-contained triangle with signed distance computation
   - All types use `alignas` for SIMD optimization (16/32/64 bytes depending on CPU)

2. **Implicit Functions** (`EBGeometry_ImplicitFunction.hpp`)
   - Abstract base class with `value(Vec3) -> Real` method
   - Convention: negative = inside, positive = outside
   - Supports polymorphic composition for complex geometries
   - GPU-compatible via `EBGEOMETRY_GPU_HOST_DEVICE` macros

3. **Analytic SDFs** (`EBGeometry_AnalyticDistanceFunctions.hpp`)
   - Built-in primitives: `PlaneSDF`, spheres, boxes
   - All derive from `ImplicitFunction`

4. **CSG Operations** (`EBGeometry_CSG.hpp`)
   - `UnionIF`: Boolean union via `min(f1->value(x), f2->value(x))`
   - Composable: can nest unions for complex shapes
   - Factory functions: `createImpFunc<Type, MemoryLocation>(args)` for host/device placement

5. **DCEL Mesh** (`EBGeometry_DCEL_*.hpp`)
   - Doubly-connected edge list for topological mesh representation
   - Components: `Vertex`, `Edge` (half-edge), `Face`, `Mesh`
   - Non-owning: typically built by `MeshParser` from file data
   - Search algorithms: `Direct`, `Direct2` (exhaustive, for use with BVH acceleration)

6. **Triangle Collections** (`EBGeometry_TriangleCollection.hpp`)
   - Two memory layout specializations via `LayoutType`:
     - **AoS** (Array-of-Structs): Contiguous `Triangle<MetaData>[]`
     - **SoA** (Structure-of-Arrays): Separate arrays for vertices, normals, etc.
   - Non-owning views (lightweight, trivially copyable for GPU transfer)
   - Both implement `value(Vec3)` for signed distance queries

### GPU Architecture

- **Backend abstraction**: CUDA or HIP (mutually exclusive, selected via CMake)
- **Macros** (`EBGeometry_GPU.hpp`):
  - `EBGEOMETRY_GPU_HOST`: CPU-only
  - `EBGEOMETRY_GPU_DEVICE`: GPU-only
  - `EBGEOMETRY_GPU_GLOBAL`: Kernel function
  - `EBGEOMETRY_GPU_HOST_DEVICE`: Dual compilation
- **Memory locations**: `Host`, `Pinned`, `Unified`, `Global`
- **Factory pattern**: `createImpFunc<Type, MemLoc>(args)` handles device construction via placement new

### Performance Features

- **SIMD alignment** (`EBGeometry_Alignas.hpp`): Dynamic alignment (AVX-512: 64B, AVX: 32B, SSE2: 16B)
- **Vectorization pragmas** (`EBGeometry_Macros.hpp`): Compiler-specific hints via `EBGEOMETRY_PRAGMA_SIMD`
- **Inlining control**: `EBGEOMETRY_ALWAYS_INLINE`, `EBGEOMETRY_INLINE`
- **Aliasing**: `EBGEOMETRY_RESTRICT` for pointer optimization
- **Compile flags**: `-O3 -march=native` for auto-vectorization

### Testing Infrastructure

- **Framework**: Catch2 v3 (requires `find_package(Catch2 3 REQUIRED)`)
- **Test location**: `Exec/Tests/*.cpp`
- **Test files**:
  - `VecTests.cpp`: Vector operations
  - `TriangleTests.cpp`: Triangle geometry and SDF
  - `VertexTests.cpp`: DCEL vertex operations
  - `AnalyticDistanceFunctionsTests.cpp`: SDF validation
  - `VirtualFunctionTests.cpp`: Polymorphic implicit functions
  - `SFCTests.cpp`: Space-filling curves
- **Running strategy**: Build individual executables, run via CTest or directly
- **Test naming**: `TEST_CASE("Component_Feature")`

## Development Workflow

### Adding New Implicit Functions
1. Create declaration in `Source/EBGeometry_NewFunction.hpp`
2. Create implementation in `Source/EBGeometry_NewFunctionImplem.hpp`
3. Include implementation at end of declaration file
4. Derive from `ImplicitFunction` base class
5. Implement `value(Vec3) const` method with signed distance logic
6. Add `EBGEOMETRY_GPU_HOST_DEVICE` to methods for GPU support
7. Include new header in `EBGeometry.hpp`
8. Add tests in `Exec/Tests/NewFunctionTests.cpp`

### Working with Templates
- `MetaData` template parameter: User-defined data attached to geometry primitives
- `LayoutType`: Compile-time selection between AoS/SoA memory layouts
- `Real` type: `float` or `double` (controlled by `ENABLE_DOUBLE` CMake option)

### GPU Development
- Use `EBGEOMETRY_GPU_HOST_DEVICE` for dual-compilation
- Factory functions manage memory location transparently
- Non-owning views (Span, TriangleCollection) are trivially copyable for device transfer
- Test GPU code paths with `ENABLE_CUDA=ON` or `ENABLE_HIP=ON`

### Code Style
- **Format**: clang-format configuration in `.clang-format`
- **Linting**: clang-tidy configuration in `.clang-tidy`
- **Pre-commit hooks**: Runs clang-format, clang-tidy, doxygen, REUSE license checks
- **License headers**: All files require SPDX headers (enforced by `reuse` tool)

## Common Development Patterns

### Memory Layout Selection
```cpp
// AoS layout: cache-friendly for single-triangle access
TriangleCollection<AoS, MetaData> aos_collection(triangles);

// SoA layout: better for SIMD across multiple triangles
TriangleCollection<SoA, MetaData> soa_collection(vertices, normals);
```

### CSG Composition
```cpp
auto sphere = createImpFunc<SphereSDF, Host>(center, radius);
auto plane = createImpFunc<PlaneSDF, Host>(point, normal);
auto union_shape = createImpFunc<UnionIF, Host>(sphere, plane);
```

### GPU Placement
```cpp
// Host memory (CPU-only)
auto host_func = createImpFunc<SphereSDF, Host>(args);

// Device global memory (GPU-only)
auto device_func = createImpFunc<SphereSDF, Global>(args);

// Unified memory (CPU/GPU shared)
auto unified_func = createImpFunc<SphereSDF, Unified>(args);
```

## File Organization

- `EBGeometry.hpp`: Main include file
- `Source/`: All library implementation
- `Exec/Tests/`: Test suite (Catch2)
- `Exec/Examples/`: Example programs
- `Docs/`: Doxygen documentation configuration
- `CMakeLists.txt`: Top-level build configuration

## Build Configuration Reference

| CMake Option | Default | Description |
|--------------|---------|-------------|
| `ENABLE_DEBUG` | `ON` | Enable debug assertions (`EBGEOMETRY_EXPECT`) |
| `ENABLE_TESTS` | `ON` | Build test executables |
| `ENABLE_EXAMPLES` | `OFF` | Build example programs |
| `ENABLE_CUDA` | `ON` | Enable CUDA GPU support |
| `ENABLE_HIP` | `OFF` | Enable HIP GPU support |
| `ENABLE_DOUBLE` | `OFF` | Use double precision (default: float) |

## Key Conventions

- **Signed distance**: Negative inside, positive outside
- **Non-owning views**: TriangleCollection, Span do not manage lifetime
- **GPU compatibility**: Mark shared functions with `EBGEOMETRY_GPU_HOST_DEVICE`
- **SIMD alignment**: Use `alignas(EBGEOMETRY_ALIGNMENT)` for data structures
- **Template specialization**: Prefer compile-time selection over runtime polymorphism
