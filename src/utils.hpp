
#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <sys/time.h>
#include <cstddef>
#include <cstdlib>

#ifndef dabs
  #define dabs(a) ((a) > 0.0 ? (a) : -(a))
#endif

double dclock();
void generate_vector_double( size_t m, double *V );

#endif
