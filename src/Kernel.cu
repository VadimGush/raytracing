#include <iostream>
#include <fstream>
#include <glm/vec3.hpp>
#include <glm/geometric.hpp>
#include <cuda_runtime_api.h>
#include <curand_mtgp32_kernel.h>
#include <device_launch_parameters.h>

#include "Display.h"
#include "RayTracer.cuh"

using namespace std;

constexpr int display_width = 1920;
constexpr int display_height = 1080;

int main() {
    ofstream image{"image.ppm", ios::out};
    if (!image) {
        cout << "Creating image file failed" << endl;
        return -1;
    }

    try {
        Display display{display_width, display_height};

        dim3 threads(16, 16);
        dim3 blocks(display_width / threads.x + 1, display_height / threads.y + 1);

        RayTracer::RenderScreen<<<blocks, threads>>>(display.GetDisplay(), display_width, display_height);
        image << display;
    } catch (const exception& e) {
        cout << "ERROR: " << e.what() << endl;
    }

    return 0;
}
