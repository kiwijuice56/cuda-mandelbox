#ifndef RAYMARCHER_FRACTALS_GPU_CAMERA_CUH
#define RAYMARCHER_FRACTALS_GPU_CAMERA_CUH

#include "shape/shape.cuh"
#include "shape/torus/torus.cuh"
#include "shape/mandelbox/mandelbox.cuh"
#include "shape/sphere/sphere.cuh"

class camera {

public:
    __host__ void render() const;

private:
    const double WIDTH = 2048;
    const double HEIGHT = 2048;
    const double FOV = 0.994;
    const double CANVAS_OFFSET = 0.005;
    const double MAX_DISTANCE = 3000;
    const int MAX_ITER = 512;
    const double THRESHOLD = 0.001;
    const double WORLD_SIZE = 12.0;
    const char *OUT_PATH = R"(C:\Users\Eric Alfaro\CLionProjects\raymarcher_fractals_gpu\out.png)";
};

#endif //RAYMARCHER_FRACTALS_GPU_CAMERA_CUH
