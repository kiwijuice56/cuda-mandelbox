#include "sphere.cuh"

__device__ double sphere::distance(double x, double y, double z) {
    return sqrt(x*x + y*y + z*z) - 1.5;
}

__device__ void sphere::norm(double x, double y, double z, double *nX, double *nY, double *nZ) {
    *nX = x;
    *nY = y;
    *nZ = z;
}
