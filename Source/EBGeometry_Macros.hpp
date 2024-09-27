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

#endif
