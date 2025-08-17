/* EBGeometry
 * Copyright © 2024 Robert Marskar
 * Please refer to Copyright.txt and LICENSE in the EBGeometry root directory.
 */

/*!
  @file   EBGeometry_Macros.hpp
  @brief  Various EBGeometry macros
  @author Robert Marskar
*/

#ifndef EBGeometry_Macros
#define EBGeometry_Macros

#include <cassert>
#include <iostream>

/*! \cond */

static unsigned long long int EBGEOMETRY_ASSERTION_FAILURES = 0;

#define EBGEOMETRY_ALWAYS_EXPECT(cond)                                                     \
  if (!(cond)) {                                                                           \
    printf("Expectation '%s' failed on line %i in file %s!\n", #cond, __LINE__, __FILE__); \
  }

// Debugging macrros.
#ifdef EBGEOMETRY_ENABLE_DEBUG
#define EBGEOMETRY_EXPECT(cond) EBGEOMETRY_ALWAYS_EXPECT(cond)
#else
#define EBGEOMETRY_EXPECT(cond) (void)0
#endif
// End debugging macros

// Inline macros
#if defined(__CUDA_ARCH__) && defined(EBGEOMETRY_ENABLE_CUDA)
#define EBGEOMETRY_ALWAYS_INLINE __forceinline__

#elif defined(__GNUC__)
#define EBGEOMETRY_ALWAYS_INLINE inline __attribute__((always_inline))

#else
#define EBGEOMETRY_ALWAYS_INLINE inline
#endif

#define EBGEOMETRY_INLINE inline
// End inline macros

/*! \endcond */

#endif
