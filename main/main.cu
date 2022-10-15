#include <chrono>
#include <iostream>

#include "camera.cuh"

int main() {
    camera c = camera();

    std::cout << "Rendering ..." << '\n';
    auto start = std::chrono::high_resolution_clock::now();

    c.render();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Render complete: " << duration.count() << " microseconds" << '\n';

    return 0;
}