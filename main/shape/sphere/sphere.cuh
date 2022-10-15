#ifndef RAYMARCHER_FRACTALS_GPU_SPHERE_CUH
#define RAYMARCHER_FRACTALS_GPU_SPHERE_CUH

#include "../shape.cuh"

class sphere: public shape {
public:
    __device__ double distance(double x, double y, double z) override;

    __device__ void norm(double x, double y, double z, double *nX, double *nY, double *nZ) override;
};


#endif //RAYMARCHER_FRACTALS_GPU_SPHERE_CUH
