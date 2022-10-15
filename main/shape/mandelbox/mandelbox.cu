#include "mandelbox.cuh"

// DE algorithm from http://blog.hvidtfeldts.net/index.php/2011/11/distance-estimated-3d-fractals-vi-the-mandelbox/

__device__ double mandelbox::distance(double x, double y, double z) {
    double oX = x, oY = y, oZ = z;
    double *zX = &x, *zY = &y, *zZ = &z;

    double dz = 1.0;
    double *dr = &dz;

    for (int i = 0; i < 32; i++) {
        boxFold(zX, zY, zZ);
        sphereFold(zX, zY, zZ, dr);

        *zX = 2.4 * *zX + oX;
        *zY = 2.4 * *zY + oY;
        *zZ = 2.4 * *zZ + oZ;

        *dr = *dr * 2.4  + 1.0;
    }
    double r = sqrt(*zX * *zX + *zY * *zY + *zZ * *zZ);
    return r / abs(*dr);
}

__device__ void mandelbox::norm(double x, double y, double z, double *nX, double *nY, double *nZ) {
    *nX = 1;
    *nY = 0;
    *nZ = 0;
}

__device__ void mandelbox::sphereFold(double *x, double *y, double *z, double *dz) {
    double r2 = *x * *x + *y * *y + *z * *z;
    if (r2 < 0.25) {
        double temp = (.5 / 0.25);
        *x = *x * temp;
        *y = *y * temp;
        *z = *z * temp;
        *dz *= temp;
    } else if (r2 < .5) {
        double temp = (.5 / r2);
        *x = *x * temp;
        *y = *y * temp;
        *z = *z * temp;
        *dz *= temp;
    }
}

__device__ void mandelbox::boxFold(double *x, double *y, double *z) {
    *x = min(32.0, max(*x, -32.0)) * 2.0 - *x;
    *y = min(32.0, max(*y, -32.0)) * 2.0 - *y;
    *z = min(32.0, max(*z, -32.0)) * 2.0 - *z;
}