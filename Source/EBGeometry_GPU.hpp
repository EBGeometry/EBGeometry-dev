// SPDX-FileCopyrightText: 2025 Robert Marskar
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file   EBGeometry_GPU.hpp
 * @brief  Declaration of GPU interface with various GPU backends
 * @author Robert Marskar
 */

#ifndef EBGEOMETRY_GPU_HPP
#define EBGEOMETRY_GPU_HPP

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
#include <hip/hip_runtime.h>
#endif

#if defined(EBGEOMETRY_ENABLE_CUDA)
#define EBGEOMETRY_ENABLE_GPU
#define EBGEOMETRY_GPU_HOST __host__
#define EBGEOMETRY_GPU_DEVICE __device__
#define EBGEOMETRY_GPU_GLOBAL __global__
#define EBGEOMETRY_GPU_HOST_DEVICE __host__ __device__
#elif defined(EBGEOMETRY_ENABLE_HIP)
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
#endif

namespace EBGeometry {

  /**
   * @brief Enum for describing memory locations in host and device
   * @details
   *
   * Invalid: Unknown location
   * Host: CPU (host)
   * Pinned: Device pinned memory
   * Unified: Device unified memory
   * Global: Device global memory
   */
  enum class MemoryLocation
  {
    Invalid,
    Host,
    Pinned,
    Unified,
    Global
  };

  namespace GPU {
    /**
     * @brief Check if an object is allocated on the device or on the host. Pointer should not be null.
     * @return True if the object lives on the device and false otherwise.
     */
    template <typename T>
    EBGEOMETRY_GPU_HOST
    [[nodiscard]] EBGEOMETRY_ALWAYS_INLINE
    bool
    isDevicePointer(const T* a_ptr) noexcept
    {
      if (a_ptr == nullptr) {
        return false;
      }

#if defined(EBGEOMETRY_ENABLE_CUDA)

      cudaPointerAttributes attr{};
      const cudaError_t     st = cudaPointerGetAttributes(&attr, static_cast<const void*>(a_ptr));

      if (st != cudaSuccess) {
        return false;
      }

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 10000)
      if ((attr.type == cudaMemoryTypeDevice) || (attr.type == cudaMemoryTypeManaged)) {
        return true;
      }
      else {
        return false;
      }
#else
      if ((attr.memoryType == cudaMemoryTypeDevice) || (attr.isManaged != 0)) {
        return true;
      }
      else {
        return false;
      }
#endif // CUDART_VERSION

#elif defined(EBGEOMETRY_ENABLE_HIP)

      hipPointerAttribute_t attr{};
      const hipError_t st = hipPointerGetAttributes(&attr, const_cast<void*>(reinterpret_cast<const void*>(a_ptr)));

      if (st != hipSuccess) {
        return false;
      }

      if (attr.memoryType == hipMemoryTypeDevice) {
        return true;
      }
#if defined(HIP_VERSION_MAJOR)
      if (attr.isManaged != 0) {
        return true;
      }
#endif
      return false;

#else
      (void)a_ptr;

      return false;
#endif
    }
  } // namespace GPU
} // namespace EBGeometry

#endif
