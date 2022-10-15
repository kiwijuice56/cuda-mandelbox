#ifndef RAYMARCHER_FRACTALS_GPU_SHAPE_CUH
#define RAYMARCHER_FRACTALS_GPU_SHAPE_CUH

class shape {
public:
    __device__ virtual double distance(double x, double y, double z) = 0;

    __device__ virtual void norm(double x, double y, double z, double *nX, double *nY, double *nZ) = 0;
};


#endif //RAYMARCHER_FRACTALS_GPU_SHAPE_CUH
