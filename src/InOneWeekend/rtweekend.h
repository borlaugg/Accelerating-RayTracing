#ifndef RTWEEKEND_H
#define RTWEEKEND_H
//==============================================================================================
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <curand_kernel.h>      // For the random uniform function


// C++ Std Usings

// We can't use these on the device : (
// using std::make_shared;
// using std::shared_ptr;


// Constants

const double infinity = INFINITY;       // Changed it from std::infinity
const double pi = 3.1415926535897932385;


// Utility Functions

__device__ __host__ inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

// __device__ __host__ inline double random_double() {      // This won't work in device as it refers to std::rand()
//     // Returns a random real in [0,1).
//     return std::rand() / (RAND_MAX + 1.0);
// }

__device__ inline double random_double(curandState* state) {
    // Returns a random real number in [0, 1).
    // Generate a random number from state and return as double
    return curand_uniform(state);
}

__device__ inline double random_double(double min, double max, curandState* state) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_double(state);
}

__host__ inline double random_double(double min, double max) {
    // Returns a random real in [min,max).
    return min + (max-min)*std::rand() / (RAND_MAX + 1.0);
}


// Common Headers

#include "color.h"
#include "interval.h"
#include "ray.h"
#include "vec3.h"


#endif
