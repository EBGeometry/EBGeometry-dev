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

// Debugging macrros.
#ifdef EBGEOMETRY_ENABLE_DEBUG

// Expectation macro which prints an error message without exiting the program.
#define EBGEOMETRY_EXPECT(cond)                                                           \
  if (!(cond)) {                                                                          \
    printf("Expectation '%s' failed on line %i in file %s\n", #cond, __LINE__, __FILE__); \
  }

// Assertion macro which exits the program if the condition is violated.
#define EBGEOMETRY_ASSERT(cond) \
  if (!(cond)) {                \
    std::abort();               \
  }

#else
#define EBGEOMETRY_EXPECT(cond) (void)0
#define EBGEOMETRY_ASSERT(cond) (void)0
#endif

// Expectation macro which prints an error message without exiting the program.
#define EBGEOMETRY_ALWAYS_EXPECT(cond)                                                    \
  if (!(cond)) {                                                                          \
    printf("Expectation '%s' failed on line %i in file %s\n", #cond, __LINE__, __FILE__); \
  }

#endif
