#include <iostream>
#include <exception>
#include "utils/cuda_unique_ptr.h"
#include <fstream>
#include <vector>
#include <glm/vec3.hpp>
#include "Display.h"
#include "RayTracer.cuh"
#include "Sphere.h"

using namespace std;
using namespace glm;

constexpr int display_width = 1920;
constexpr int display_height = 1080;

ostream& operator<<(ostream& output, const vec3& vector) {
    return output << vector.x << " " << vector.y << " " << vector.z;
}

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

        vector<Sphere> spheres = {
                Sphere{ 0.3,   { 0,     0, -1 }},
                Sphere{ 10,     { 0, -10.30, -1 }},
                Sphere{ 0.2,   { 0.6, 0, -1.1}}
        };
        cuda_unique_ptr<Sphere> device_spheres(3);
        device_spheres.copy_from(spheres.data());

        RayTracer::RenderScreen<<<blocks, threads>>>(
                device_spheres.get_pointer(),
                spheres.size(),
                display.GetDisplay(), display_width, display_height
        );

        image << display;
    } catch (const exception& e) {
        cout << "ERROR: " << e.what() << endl;
    }

    return 0;
}
