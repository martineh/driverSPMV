#include "utils.hpp"

double dclock() {
  struct timeval  tv;
  // struct timezone tz;

  gettimeofday( &tv, NULL );

  return (double) (tv.tv_sec + tv.tv_usec*1.0e-6);
}


void generate_vector_double( size_t m, double *V ) {
  for ( size_t i=0; i<m; i++ ) V[i] = ((double) rand())/RAND_MAX + 1.0;
}
