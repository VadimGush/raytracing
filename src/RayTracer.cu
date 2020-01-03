//
// Created by Vadim Gush on 27.12.2019.
//

#include "RayTracer.cuh"

#include <glm/geometric.hpp>
#include <cstdlib>
#include "Camera.h"

#define RAY_COUNT 50
#define BACKGROUND vec3{0.8,0.9,1}
#define LAST_RAY_COLOR vec3{0,0,0}
#define PI 3.14159265359f;

using namespace glm;

struct Hit {
    float distance;
    vec3 normal;
    vec3 position;
};

struct LocalRandom {
    float* random_numbers;
    const int max;
};

__device__ inline float RandomFloat(const LocalRandom& random, int& random_id) {
    random_id+=10;
    if (random_id >= random.max) random_id = 0;
    return random.random_numbers[random_id];
}

__device__ Hit HitSphere(const Sphere& sphere, const Camera& camera) {
    vec3 oc = camera.origin - sphere.position;

    float a = dot(camera.direction, camera.direction);
    float b = 2 * dot(oc, camera.direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;

    float d = b * b - 4 * a * c;

    if (d > 0) {
        float x1 = (-b - sqrt(d)) / (2 * a);
        float x2 = (-b + sqrt(d)) / (2 * a);

        float mt = min(x1, x2);

        vec3 hit_position = camera.origin + camera.direction * mt;
        return {mt, (hit_position - sphere.position) / sphere.radius, hit_position};
    }
    return {-1};
}

__device__ inline vec3 RandomVector(const LocalRandom random, int& random_id) {
    float theta = RandomFloat(random, random_id) * 2 * PI;
    float z = RandomFloat(random, random_id) * 2 - 1;
    float temp = sqrt(1 - z * z);
    return {temp * cos(theta), temp * sin(theta), z};
}

__device__ vec3 Render(const CUDA::device_ptr<Sphere>& spheres, Camera camera, int iter, const LocalRandom local_random, int& random_id) {

    bool is_background = true;
    vec3 current_color = vec3{1,1,1};
    float k = 2;

    const int rays = 10;
    for (int ray_id = 0; ray_id < rays; ray_id++) {

        // register hit
        Hit hit{-1};
        Sphere current_sphere;
        for (int i = 0; i < spheres.size(); i++) {
            const Sphere& sphere = spheres.get()[i];
            Hit current = HitSphere(sphere, camera);
            if (current.distance < 0.0001f) continue;
            if (current.distance < hit.distance || hit.distance == -1) {
                current_sphere = sphere;
                hit = current;
            }
        }

        const Material& material = current_sphere.material;
        if (hit.distance > 0) {

            is_background = false;
            camera.origin = hit.position;

            if (material.diel) {
                vec3 outward_normal;
                float ni_over_nt;
                if (dot(camera.direction, hit.normal) > 0) {
                    outward_normal = -hit.normal;
                    ni_over_nt = material.refractive;
                } else {
                    outward_normal = hit.normal;
                    ni_over_nt = 1.0f / material.refractive;
                }

                camera.direction = refract(camera.direction, outward_normal, ni_over_nt);
                continue;
            }

            // This only for metal and scatter materials
            k /= 2;
            current_color *= material.color;

            if (material.light)
                break;

            if (material.metal) {
                camera.direction = reflect(camera.direction, hit.normal);
                camera.direction += RandomVector(local_random, random_id) * (1.0f - material.reflect);
                camera.direction = normalize(camera.direction);
                continue;
            }

            // if scatter
            camera.direction = normalize(hit.normal + RandomVector(local_random, random_id));
        } else if (!is_background) {
            k /= 2;
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

            color += Render(spheres, Camera{{0,0,0},{x, y, -1}}, 0, random, random_id);
        }
        pixel = color / (float)RAY_COUNT;
    }
}
