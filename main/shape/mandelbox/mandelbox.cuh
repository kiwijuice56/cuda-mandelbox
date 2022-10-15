#ifndef RAYMARCHER_FRACTALS_GPU_MANDELBOX_CUH
#define RAYMARCHER_FRACTALS_GPU_MANDELBOX_CUH

#include "../shape.cuh"

class mandelbox: public shape {
public:
    __device__ double distance(double x, double y, double z) override;

    __device__ void norm(double x, double y, double z, double *nX, double *nY, double *nZ) override;

    static __device__ void boxFold(double *x, double *y, double *z);

    static __device__ void sphereFold(double *x, double *y, double *z, double *dz);
};

#endif //RAYMARCHER_FRACTALS_GPU_MANDELBOX_CUH
