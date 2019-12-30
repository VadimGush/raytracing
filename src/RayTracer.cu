//
// Created by Vadim Gush on 27.12.2019.
//

#include "RayTracer.cuh"

#include <glm/geometric.hpp>
#include <cstdlib>

#define RAY_COUNT 50

using namespace glm;

struct Hit {
    float distance;
    vec3 color;
    vec3 normal;
};

struct LocalRandom {
    float* random_numbers;
    const int max;
};

__device__ float RandomFloat(const LocalRandom& random, int& random_id) {
    random_id++;
    if (random_id >= random.max) random_id = 0;
    return random.random_numbers[random_id];
}

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

__device__ vec3 RandomVector(const LocalRandom random, int& random_id) {
    return {
        RandomFloat(random, random_id) * 2 - 1,
        RandomFloat(random, random_id) * 2 - 1,
        RandomFloat(random, random_id) * 2 - 1
    };
}

__device__ vec3 Render(Sphere* spheres, const int spheres_count, const vec3& camera_origin, const vec3& camera_direction, int iter, const LocalRandom local_random, int& random_id) {

    Hit hit{-1, {0,0,0}, {0,0,0}};
    Sphere current_sphere;
    for (int i = 0; i < spheres_count; i++) {
        Sphere& sphere = spheres[i];
        Hit current = HitSphere(sphere.position, sphere.radius, camera_origin, camera_direction);

        if (hit.distance <= 0 || (hit.distance > current.distance && current.distance > 0)) {
            current_sphere = sphere;
            hit = current;
        }
    }

    if (hit.distance > 0.0001) {

        vec3 point = camera_origin + camera_direction * hit.distance;

        iter++;
        if (iter < 6) {
            if (!current_sphere.light) {
                return 0.5f * current_sphere.color * Render(spheres, spheres_count, point, normalize(hit.normal + RandomVector(local_random, random_id)), iter, local_random, random_id);
            } else {
                return current_sphere.color;
            }
        } else {
            return {0,0,0};
        }
    } else {
        return {0.1,0.1,0.1};
    }
}


__global__ void RayTracer::RenderScreen(
        CUDA::device_ptr<Sphere> spheres,
        const int spheres_count,
        CUDA::device_ptr<float> random_numbers,
        const int numbers_per_thread,
        CUDA::device_ptr<vec3> display_ptr,
        const int display_width,
        const int display_height) {

    vec3* display = display_ptr.get();
    int xi = threadIdx.x + blockDim.x * blockIdx.x;
    int yi = threadIdx.y + blockDim.y * blockIdx.y;
    vec3& pixel = display[xi + yi * display_width];

    LocalRandom random{
        &random_numbers.get()[(xi + yi * display_width) * numbers_per_thread],
        numbers_per_thread
    };
    int random_id = 0;

    if (xi < display_width && yi < display_height) {

        vec3 color{0,0,0};
        for (int i = 0; i < RAY_COUNT; i++) {

            float x = ((float)xi + RandomFloat(random, random_id)) / display_width - 0.5;
            float y = ((float)yi + RandomFloat(random, random_id)) / display_height - 0.5;
            float aspect = (float) display_width / display_height;
            x *= aspect;

            color += Render(spheres.get(), spheres_count, {0,0,0}, {x, y, -1}, 0, random, random_id);
        }
        pixel = color / (float)RAY_COUNT;

        // debug
        // if (threadIdx.x == 0 || threadIdx.y == 0) pixel.b += 0.5;
        // if (x == 0 || y == 0) pixel.r += 0.5;
    }
}
