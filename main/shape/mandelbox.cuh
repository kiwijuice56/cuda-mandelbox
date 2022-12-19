#ifndef RAYMARCHER_FRACTALS_GPU_MANDELBOX_CUH
#define RAYMARCHER_FRACTALS_GPU_MANDELBOX_CUH

class mandelbox {
public:
    __device__ double distance(double x, double y, double z);

    static __device__ void boxFold(double *x, double *y, double *z);

    static __device__ void sphereFold(double *x, double *y, double *z, double *dz);

public:
    double renderScale = 1.7;
};

#endif //RAYMARCHER_FRACTALS_GPU_MANDELBOX_CUH
