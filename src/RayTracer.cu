//
// Created by Vadim Gush on 27.12.2019.
//

#include "RayTracer.cuh"

#include <glm/geometric.hpp>

using namespace glm;

struct Hit {
    float distance;
    vec3 color;
};

__device__ Hit HitSphere(const vec3& center, const float radius, const vec3& origin, const vec3& direction) {
    vec3 oc = origin - center;
    float a = dot(direction, direction);
    float b = 2 * dot(oc, direction);
    float c = dot(oc, oc) - radius * radius;
    float d = b * b - 4 * a * c;
    if (d < 0) {
        return {-1, {}};
    } else {
        float t = (-b - sqrt(d)) / (2*a);
        vec3 hit_position = origin + direction * t;
        vec3 sphere_normal = hit_position - center;

        return {length(hit_position - origin), normalize(sphere_normal)};
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

        // some pre-calculations
        float x = (float) xi / display_width - 0.5;
        float y = (float) yi / display_height - 0.5;
        float aspect = (float) display_width / display_height;
        x *= aspect;

        vec3 direction{x, y, -1};
        vec3 origin{0,0,0};

        vec3 sphere_center = vec3{0,0,-1};
        Hit hit = HitSphere(sphere_center, 0.3, origin, direction);
        if (hit.distance > 0)
            pixel = hit.color;

        // debug
        // if (threadIdx.x == 0 || threadIdx.y == 0) pixel.b += 0.5;
        // if (x == 0 || y == 0) pixel.r += 0.5;
    }
}
