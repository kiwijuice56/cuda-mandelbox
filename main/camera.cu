#include <iostream>
#include "camera.cuh"

__global__ static void cast_ray(sf::Uint8 *pixels, mandelbox s,
                                double VIEW_WIDTH, double VIEW_HEIGHT,
                                double WIDTH, double HEIGHT, double CANVAS_OFFSET,
                                double CAMX, double CAMY, double CAMZ,
                                double MAX_DISTANCE, int MAX_ITER, double THRESHOLD, double WORLD_SIZE) {

    // Get the position on the viewport by getting the pixel first
    unsigned int ord = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = (ord / (int) WIDTH);
    unsigned int col = (ord % (int) HEIGHT);

    double pxX = (col / WIDTH) * VIEW_WIDTH - (VIEW_WIDTH / 2);
    double pxY = (row / HEIGHT) * VIEW_HEIGHT - (VIEW_HEIGHT / 2);

    // Start at the camera position
    double rayPX = CAMX, rayPY = CAMY, rayPZ = CAMZ;
    double rayDX = pxX, rayDY = pxY, rayDZ = CANVAS_OFFSET;
    double raySize = sqrt(rayDX*rayDX + rayDY*rayDY + rayDZ*rayDZ);

    // Normalize the direction vector
    rayDX /= raySize;
    rayDY /= raySize;
    rayDZ /= raySize;

    // March ray until threshold is reached
    double distance, totalDistance = 0;
    for (int i = 0; i < MAX_ITER; i++) {
        // Loops the world space
        rayPX = remainder(rayPX, WORLD_SIZE);
        rayPY = remainder(rayPY, WORLD_SIZE);
        rayPZ = remainder(rayPZ, WORLD_SIZE);

        distance = s.distance(rayPX, rayPY, rayPZ);
        totalDistance += distance;

        if (abs(totalDistance) > MAX_DISTANCE)
            break;

        // Color the pixel when the ray is close enough and break
        if (abs(distance) < THRESHOLD) {
            double complexity = 1.0 - (i / (double) MAX_ITER);
            double distance_fade = 1.0 - pow(totalDistance / MAX_DISTANCE, 2);

            auto baseColor = 255 - (unsigned char) (255 * complexity * distance_fade);
            int p = (int) (row * 4 * WIDTH + 4 * col);
            pixels[p++] = baseColor;
            pixels[p++] = baseColor;
            pixels[p++] = baseColor;
            pixels[p++] = 255;
            return;
        }

        // Move the ray forward
        rayPX += rayDX * distance;
        rayPY += rayDY * distance;
        rayPZ += rayDZ * distance;
    }

    // Default background color
    int p = (int) (row * 4 * WIDTH + 4 * col);
    pixels[p++] = 255;
    pixels[p++] = 255;
    pixels[p++] = 255;
    pixels[p++] = 255;
}

__host__ void camera::render(sf::Uint8 *pixels, mandelbox drawnShape) const {
    // Compute the view plane's width and height
    const double VIEW_WIDTH = 2 * tan(FOV / 2) * CANVAS_OFFSET;
    const double VIEW_HEIGHT = VIEW_WIDTH * (HEIGHT / WIDTH);

    // Create threads
    cast_ray<<<8192, 32>>>(pixels, drawnShape, VIEW_WIDTH, VIEW_HEIGHT, WIDTH, HEIGHT, CANVAS_OFFSET,
                           posX, posY, -posZ,
                           MAX_DISTANCE, MAX_ITER, THRESHOLD, WORLD_SIZE);

    cudaDeviceSynchronize();
}