#include <iostream>
#include <exception>
#include <fstream>
#include <vector>
#include <glm/vec3.hpp>
#include <curand.h>
#include "utils/cuda_memory.h"
#include "utils/logger.h"
#include "utils/png.h"
#include "Display.h"
#include "RayTracer.h"
#include "Sphere.h"

using namespace std;
using namespace glm;

constexpr int display_width = 1920;
constexpr int display_height = 1080;

ostream& operator<<(ostream& output, const vec3& vector) {
    return output << vector.x << " " << vector.y << " " << vector.z;
}

CUDA::unique_ptr<float> GenerateRandomNumbers(const size_t size) {
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, 123);
    CUDA::unique_ptr<float> device_random_numbers(size);
    curandGenerateUniform(
            generator,
            device_random_numbers.get_device_pointer().get(),
            device_random_numbers.get_device_pointer().size());
    return move(device_random_numbers);
}

CUDA::unique_ptr<Sphere> CreateWorld() {
    vector<Sphere> spheres = {
            Sphere{ 0.3,   { 0,     0, -1}      , Material::Scatter({1,1,1})}, // big red sphere
            Sphere{ 20,    { 0, -20.30, -1}     , Material::Scatter({0.1,1,0.1})}, // floor
            Sphere{ 0.2,   { 0.6, 0, -1.1}      , Material::Metal({1,1,0}, 0.1)},
            Sphere{ 0.06,  { 0.35,-0.2, -1.1}   , Material::Light({1,1,1})},
            Sphere{ 0.02,  { 0.35,-0.25, -0.8}  , Material::Light({1,0,1})},
            Sphere{ 0.05,  { 0.25,-0.25, -0.9}  , Material::Metal({1,1,1}, 1)},
            Sphere{ 0.04,  { 0.15,-0.25, -0.8}  , Material::Light({0,1,1})},
            Sphere{ 0.1,   { 0.5, -0.3, -0.9}   , Material::Metal({1,1,1}, 0.9)},
            Sphere{ 0.04,  { 0.3, -0.27, -0.7}   , Material::Dielectric(1.5f)},
    };
    CUDA::unique_ptr<Sphere> device_spheres(spheres.size());
    device_spheres.copy_from(spheres.data());
    return move(device_spheres);
}

int main() {

    try {
        Display display{display_width, display_height};

        const auto world = CreateWorld();
        const auto random_numbers = GenerateRandomNumbers(display_width * display_height);

        dim3 threads(16, 16);
        dim3 blocks(display_width / threads.x + 1, display_height / threads.y + 1);

        Logger::info() << "Rendering" << endl;

        // RENDER
        RayTracer::RenderScreen<<<blocks, threads>>>(
                world.get_device_pointer(),
                random_numbers.get_device_pointer(),
                display.GetDisplay(), display_width, display_height
        );
        const auto image = display.GetImage();

        Logger::info() << "Exporting image to PNG" << endl;
        PNG::WriteImage("output.png", display_width, display_height, image);

        Logger::info() << "Done!" << endl;
    } catch (const exception& e) {
        Logger::fatal() << e.what() << endl;
    }

    return 0;
}
