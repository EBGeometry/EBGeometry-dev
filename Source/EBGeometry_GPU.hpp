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

#if defined EBGEOMETRY_CUDA
#include "cuda.h"
#elif defined EBGEOMETRY_HIP
#endif

#endif
