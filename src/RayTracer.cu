//
// Created by Vadim Gush on 27.12.2019.
//

#include "RayTracer.cuh"

#include <glm/geometric.hpp>
#include <cstdlib>

#define RAY_COUNT 50
#define BACKGROUND vec3{0.01,0.01,0.01}
#define LAST_RAY_COLOR vec3{0,0,0}

using namespace glm;

struct Hit {
    float distance;
    vec3 normal;

    __device__ inline bool WasHit() const {
        return distance > 0.000001;
    }
};

struct LocalRandom {
    float* random_numbers;
    const int max;
};

__device__ inline float RandomFloat(const LocalRandom& random, int& random_id) {
    random_id++;
    if (random_id >= random.max) random_id = 0;
    return random.random_numbers[random_id];
}

__device__ inline vec3 reflect(const vec3 vector, const vec3 normal) {
    return vector - 2 * dot(vector, normal) * normal;
}

__device__ Hit HitSphere(const vec3& center, const float radius, const vec3& origin, const vec3& direction) {
    vec3 oc = origin - center;
    float a = dot(direction, direction);
    float b = 2 * dot(oc, direction);
    float c = dot(oc, oc) - radius * radius;
    float d = b * b - 4 * a * c;
    if (d < 0) {
        return {-1};
    } else {
        float t = (-b - sqrt(d)) / (2*a);
        vec3 hit_position = origin + direction * t;
        vec3 sphere_normal = hit_position - center;

        return {t, normalize(sphere_normal)};
    }
}

__device__ inline vec3 RandomVector(const LocalRandom random, int& random_id) {
    return {
        RandomFloat(random, random_id) * 2 - 1,
        RandomFloat(random, random_id) * 2 - 1,
        RandomFloat(random, random_id) * 2 - 1
    };
}

__device__ vec3 Render(const CUDA::device_ptr<Sphere>& spheres, vec3 camera_origin, vec3 camera_direction, int iter, const LocalRandom local_random, int& random_id) {

    bool is_background = true;
    vec3 current_color = vec3{1,1,1};
    float k = 2;

    for (int ray_id = 0; ray_id < 10; ray_id++) {
        // register hit
        Hit hit{-1};
        Sphere current_sphere;
        for (int i = 0; i < spheres.size(); i++) {
            const Sphere& sphere = spheres.get()[i];
            Hit current = HitSphere(sphere.position, sphere.radius, camera_origin, camera_direction);
            if (hit.distance <= 0 || (hit.distance > current.distance && current.distance > 0)) {
                current_sphere = sphere;
                hit = current;
            }
        }

        const Material& material = current_sphere.material;
        k /= 2;
        if (hit.WasHit()) {
            // hit is now
            is_background = false;
            current_color *= material.color;

            if (material.light)
                break;

            // if metal
            if (material.metal) {
                camera_origin = camera_origin + camera_direction * hit.distance;
                camera_direction = reflect(camera_direction, hit.normal);
                camera_direction += RandomVector(local_random, random_id) * material.reflect;
                continue;
            }

            // if scatter
            camera_origin = camera_origin + camera_direction * hit.distance;
            camera_direction = normalize(hit.normal + RandomVector(local_random, random_id));

        } else if (!is_background) {
            // there was hit before but now we didn't hit anything
            current_color *= BACKGROUND;
            break;
        } else {
            // hit never occur
            break;
        }
    }

    if (is_background) {
        return BACKGROUND;
    } else {
        return k * current_color;
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

            color += Render(spheres, {0,0,0}, {x, y, -1}, 0, random, random_id);
        }
        pixel = color / (float)RAY_COUNT;

        // renders CUDA blocks grid on screen
        // if (threadIdx.x == 0 || threadIdx.y == 0) pixel.b += 0.5;
    }
}
