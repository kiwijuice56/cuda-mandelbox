#ifndef RAYMARCHER_FRACTALS_GPU_CAMERA_CUH
#define RAYMARCHER_FRACTALS_GPU_CAMERA_CUH

#include <SFML/Config.hpp>
#include "shape/mandelbox.cuh"



class camera {

public:
    __host__ void render(sf::Uint8 *pixels, mandelbox drawnShape) const;

public:
    const double WIDTH = 512;
    const double HEIGHT = 512;
    const double FOV = 0.994;
    const double CANVAS_OFFSET = 0.005;
    const double MAX_DISTANCE = 32;
    const int MAX_ITER = 64;
    const double THRESHOLD = 0.01;
    const double WORLD_SIZE = 12.0;

    double posX{}, posY{}, posZ = -6;
};

#endif //RAYMARCHER_FRACTALS_GPU_CAMERA_CUH
