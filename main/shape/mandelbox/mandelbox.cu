#include "mandelbox.cuh"

// DE algorithm from http://blog.hvidtfeldts.net/index.php/2011/11/distance-estimated-3d-fractals-vi-the-mandelbox/

__device__ double mandelbox::distance(double x, double y, double z) {
    double oX = x, oY = y, oZ = z;
    double *zX = &x, *zY = &y, *zZ = &z;

    double dz = 1.0;
    double *dr = &dz;

    for (int i = 0; i < 7; i++) {
        boxFold(zX, zY, zZ);
        sphereFold(zX, zY, zZ, dr);

        *zX = 3.1 * *zX + oX;
        *zY = 3.1 * *zY + oY;
        *zZ = 3.1 * *zZ + oZ;

        *dr = *dr * 3.1 + 1.0;
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
    if (r2 < 0.5) {
        double temp = (1.5 / 0.5);
        *x = *x * temp;
        *y = *y * temp;
        *z = *z * temp;
        *dz *= temp;
    } else if (r2 < 1.5) {
        double temp = (1.5 / r2);
        *x = *x * temp;
        *y = *y * temp;
        *z = *z * temp;
        *dz *= temp;
    }
}

__device__ void mandelbox::boxFold(double *x, double *y, double *z) {
    *x = min(15.0, max(*x, -15.0)) * 2.0 - *x;
    *y = min(15.0, max(*y, -15.0)) * 2.0 - *y;
    *z = min(15.0, max(*z, -15.0)) * 2.0 - *z;
}