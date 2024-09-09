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

#include <iostream>

// Debugging macrros.
#ifdef EBGEOMETRY_DEBUG

// Expectation macro which prints an error message without exiting the program.
#define EBGEOMETRY_EXPECT(cond)                                                                            \
  if (!(cond)) {                                                                                           \
    std::cerr << __FILE__ << " (L" << __LINE__ << "): Expectation '" << #cond << "' failed!" << std::endl; \
  }

// Assertion macro which exits the program if the condition is violated.
#define EBGEOMETRY_ASSERT(cond)                                                                          \
  if (!(cond)) {                                                                                         \
    std::cerr << __FILE__ << " (L" << __LINE__ << "): Assertion '" << #cond << "' failed!" << std::endl; \
    std::exit(-1);                                                                                       \
  }
#else
#define EBGEOMETRY_EXPECT(cond) (void)0
#define EBGEOMETRY_ASSERT(cond) (void)0
#endif

#endif
