#include <iostream>
#include <exception>
#include "utils/cuda_unique_ptr.h"
#include <fstream>
#include <vector>
#include <glm/vec3.hpp>
#include <curand.h>
#include "Display.h"
#include "RayTracer.cuh"
#include "Sphere.h"

using namespace std;
using namespace glm;

constexpr int display_width = 1920;
constexpr int display_height = 1080;
constexpr int numbers_per_thread = 300;

ostream& operator<<(ostream& output, const vec3& vector) {
    return output << vector.x << " " << vector.y << " " << vector.z;
}

CUDA::unique_ptr<float> generateRandomNumbers() {
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, 123);
    CUDA::unique_ptr<float> device_random_numbers(display_width * display_height * numbers_per_thread);
    curandGenerateUniform(
            generator,
            device_random_numbers.get_device_pointer().get(),
            device_random_numbers.get_device_pointer().size());
    return move(device_random_numbers);
}

int main() {

    try {
        Display display{display_width, display_height};

        dim3 threads(16, 16);
        dim3 blocks(display_width / threads.x + 1, display_height / threads.y + 1);

        // SPHERES
        vector<Sphere> spheres = {
                Sphere{ 0.3,   { 0,     0, -1}      , Material::Scatter({1,1,1})}, // big red sphere
                Sphere{ 10,    { 0, -10.30, -1}     , Material::Scatter({0.8,1,0.8})}, // floor
                Sphere{ 0.2,   { 0.6, 0, -1.1}      , Material::Metal({1,1,0}, 0.5)},
                Sphere{ 0.06,  { 0.35,-0.2, -1.1}   , Material::Light({1,1,1})},
                Sphere{ 0.02,  { 0.35,-0.25, -0.8}  , Material::Light({1,0,1})},
                Sphere{ 0.04,  { 0.15,-0.25, -0.8}  , Material::Light({0,1,1})},
                Sphere{ 0.1,   { 0.5, -0.3, -0.9}   , Material::Metal({1,1,1}, 0.1)},
        };
        for (auto& sphere : spheres) {
            sphere.position.x -= 0.25;
        }
        CUDA::unique_ptr<Sphere> device_spheres(spheres.size());
        device_spheres.copy_from(spheres.data());

        // RANDOM GENERATOR
        CUDA::unique_ptr<float> device_random_numbers = generateRandomNumbers();

        // RENDER
        RayTracer::RenderScreen<<<blocks, threads>>>(
                device_spheres.get_device_pointer(),
                spheres.size(),
                device_random_numbers.get_device_pointer(),
                numbers_per_thread,
                display.GetDisplay(), display_width, display_height
        );

        ofstream image{"image.ppm", ios::out};
        if (!image) {
            cout << "Creating image file failed" << endl;
            return -1;
        }
        image << display;
    } catch (const exception& e) {
        cout << "ERROR: " << e.what() << endl;
    }

    return 0;
}
