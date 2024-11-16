/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_GPU.hpp
  @brief  Declaration of GPU interface with various GPU backends
  @author Robert Marskar
*/

#ifndef EBGeometry_GPU
#define EBGeometry_GPU

// Our includes
#include "EBGeometry_Macros.hpp"

// Can only define one GPU backend
#if defined(EBGEOMETRY_ENABLE_CUDA) && defined(EBGEOMETRY_ENABLE_HIP)
#error "Can not define both EBGEOMETRY_ENABLE_CUDA and EBGEOMETRY_ENABLE_HIP"
#endif

// Include GPU library headers
#if defined(EBGEOMETRY_ENABLE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#elif defined(EBGEOMETRY_ENABLE_HIP)
#endif

// CUDA definitions
#if defined(EBGEOMETRY_ENABLE_CUDA)
#include <cuda.h>

#define EBGEOMETRY_ENABLE_GPU
#define EBGEOMETRY_GPU_HOST __host__
#define EBGEOMETRY_GPU_DEVICE __device__
#define EBGEOMETRY_GPU_GLOBAL __global__
#define EBGEOMETRY_GPU_HOST_DEVICE __host__ __device__

#else

#define EBGEOMETRY_GPU_HOST
#define EBGEOMETRY_GPU_DEVICE
#define EBGEOMETRY_GPU_GLOBAL
#define EBGEOMETRY_GPU_HOST_DEVICE

#endif // <--- End CUDA definition

enum class MemoryLocation // NOLINT
{
  Invalid,
  Host,
  Pinned,
  Unified,
  Global
};

namespace GPU {
  /*!
    @brief Check if an object is allocated on the device or on the host. Pointer should not be null.
    @return True if the object lives on the device and false otherwise.
  */
  template <typename T>
  EBGEOMETRY_GPU_HOST
  EBGEOMETRY_ALWAYS_INLINE
  bool
  isDevicePointer(const T* a_ptr) noexcept
  {
    EBGEOMETRY_ALWAYS_EXPECT(a_ptr != nullptr);

    bool livesOnDevice = false;

#if EBGEOMETRY_ENABLE_CUDA
    cudaPointerAttributes attr;

    cudaPointerGetAttributes(&attr, a_ptr);

    if ((attr.type == cudaMemoryTypeDevice) || (attr.type == cudaMemoryTypeManaged)) {
      livesOnDevice = true;
    }
#endif

    return livesOnDevice;
  }
} // namespace GPU

#endif
