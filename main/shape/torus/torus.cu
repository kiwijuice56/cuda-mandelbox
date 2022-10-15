#include "torus.cuh"

__device__ double torus::distance(double x, double y, double z) {
    double qX = sqrt(x*x + z*z) - 0.8;
    double qY = y;
    return sqrt(qX*qX + qY*qY) - 0.25;
}

__device__ void torus::norm(double x, double y, double z, double *nX, double *nY, double *nZ) {
    *nX = x;
    *nY = y;
    *nZ = z;
}
