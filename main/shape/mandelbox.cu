#include "mandelbox.cuh"

// DE algorithm from http://blog.hvidtfeldts.net/index.php/2011/11/distance-estimated-3d-fractals-vi-the-mandelbox/

__device__ double mandelbox::distance(double x, double y, double z) {
    double oX = x, oY = y, oZ = z;
    double *zX = &x, *zY = &y, *zZ = &z;

    double dz = 1.0;
    double *dr = &dz;

    for (int i = 0; i < 8; i++) {
        boxFold(zX, zY, zZ);
        sphereFold(zX, zY, zZ, dr);

        *zX = renderScale * *zX + oX;
        *zY = renderScale * *zY + oY;
        *zZ = renderScale * *zZ + oZ;

        *dr = *dr * abs(renderScale) + 1.0;
    }
    return sqrt(*zX * *zX + *zY * *zY + *zZ * *zZ) / abs(*dr);
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
    *x = min(.5, max(*x, -.5)) * 2.0 - *x;
    *y = min(.5, max(*y, -.5)) * 2.0 - *y;
    *z = min(.5, max(*z, -.5)) * 2.0 - *z;
}