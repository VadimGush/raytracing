//
// Created by Vadim Gush on 27.12.2019.
//

#include "RayTracer.cuh"

#include <glm/geometric.hpp>
#include <cstdlib>

#define RAY_COUNT 10

using namespace glm;

struct Hit {
    float distance;
    vec3 color;
    vec3 normal;
};

// Random number generator
__device__ unsigned long xorshf96(int value) {
    unsigned long x=123456789 + value, y=362436069, z=521288629;
    unsigned long t;
    x ^= x << 16;
    x ^= x >> 5;
    x ^= x << 1;
    t = x;
    x = y;
    y = z;
    z = t ^ x ^ y;
    return z;
}

class Rand {
public:
    __device__ Rand(const int seed) : seed_(seed) {}

    // generate real number in [0, 1]
    __device__ float Float() {
        seed_++;
        return static_cast<float>(xorshf96(seed_) % 10000) / 10000.0f;
    }

    // generate real number in [-1, 1]
    __device__ float FullFloat() {
        return Float() * 2 - 1;
    }
private:
    int seed_;
};

__device__ Hit HitSphere(const vec3& center, const float radius, const vec3& origin, const vec3& direction) {
    vec3 oc = origin - center;
    float a = dot(direction, direction);
    float b = 2 * dot(oc, direction);
    float c = dot(oc, oc) - radius * radius;
    float d = b * b - 4 * a * c;
    if (d < 0) {
        return {-1, {}, {}};
    } else {
        float t = (-b - sqrt(d)) / (2*a);
        vec3 hit_position = origin + direction * t;
        vec3 sphere_normal = hit_position - center;

        return {t, {1,radius,1}, normalize(sphere_normal)};
    }
}

__device__ vec3 Render(const vec3& camera_origin, const vec3& camera_direction, int iter, Rand& rand) {
    vec3 sphere_center = vec3{0, 0, -1};
    Hit target = HitSphere(sphere_center, 0.3, camera_origin, camera_direction);
    Hit floor = HitSphere({0, -2.31, -1}, 2, camera_origin, camera_direction);

    Hit hit = target;
    if (hit.distance > 0) {
        if (floor.distance > 0 && hit.distance > floor.distance)
            hit = floor;
    } else {
        hit = floor;
    }

    if (hit.distance > 0.0001) {

        vec3 random{rand.Float()*2-1, rand.Float()*2-1, rand.Float()*2-1};

        vec3 point = camera_origin + camera_direction * hit.distance;
        vec3 t = point + normalize(hit.normal + random / 3.0f);

        iter++;
        if (iter < 5)
            return 0.5f * Render(point, t - point, iter, rand);
        else
            return {0,0,0};
    } else {
        return {0.75,0.75,1};
    }
}

__global__ void RayTracer::RenderScreen(
        vec3* display,
        const int display_width,
        const int display_height) {

    int xi = threadIdx.x + blockDim.x * blockIdx.x;
    int yi = threadIdx.y + blockDim.y * blockIdx.y;
    vec3& pixel = display[xi + yi * display_width];

    if (xi < display_width && yi < display_height) {

        Rand rand((xi + display_width * yi) * 100);

        vec3 color{0,0,0};
        for (int i = 0; i < RAY_COUNT; i++) {

            float x = ((float)xi + rand.Float()) / display_width - 0.5;
            float y = ((float)yi + rand.Float()) / display_height - 0.5;
            float aspect = (float) display_width / display_height;
            x *= aspect;

            color += Render({0,0,0}, {x, y, -1}, 0, rand);
        }
        pixel = color / (float)RAY_COUNT;

        // debug
        // if (threadIdx.x == 0 || threadIdx.y == 0) pixel.b += 0.5;
        // if (x == 0 || y == 0) pixel.r += 0.5;
    }
}
