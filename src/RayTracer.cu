//
// Created by Vadim Gush on 27.12.2019.
//

#include "RayTracer.h"

#include <glm/geometric.hpp>
#include <device_launch_parameters.h>
#include "Camera.h"

#define RAY_COUNT 100
#define BACKGROUND vec3{0.8,0.8,1}
#define PI 3.14159265359f

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

__device__ float RandomFloat(const LocalRandom& random, int& random_id) {
    random_id++;
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
        float x1 = (-b - sqrtf(d)) / (2 * a);
        float x2 = (-b + sqrtf(d)) / (2 * a);

        float mt = min(x1, x2);

        vec3 hit_position = camera.origin + camera.direction * mt;
        return {mt, (hit_position - sphere.position) / sphere.radius, hit_position};
    }
    return {-1};
}

__device__ vec3 RandomVector(const LocalRandom& random, int& random_id) {
    float theta = RandomFloat(random, random_id) * 2 * PI;
    float z = RandomFloat(random, random_id) * 2 - 1;
    float temp = sqrtf(1 - z * z);
    return {temp * cosf(theta), temp * sinf(theta), z};
}

__device__ float Lick(float cosine, float refraction) {
    float r0 = (1 - refraction) / (1 + refraction);
    r0 = r0 * r0;
    return r0 + (1 - r0) * powf((1 - cosine), 5);
}

__device__ void FindSphere(const CUDA::device_ptr<Sphere>& spheres, const Camera& camera, Hit& hit, Sphere& current_sphere) {
    for (int i = 0; i < spheres.size(); i++) {
        const Sphere& sphere = spheres.get()[i];
        Hit current = HitSphere(sphere, camera);
        if (current.distance < 0.0001f) continue;
        if (current.distance < hit.distance || hit.distance == -1) {
            current_sphere = sphere;
            hit = current;
        }
    }
}

__device__ void RenderDielectric(Camera& camera, const Hit& hit, const Material& material, const LocalRandom& local_random, int& random_id) {
    vec3 outward_normal;
    float ni_over_nt;
    float cosine;

    if (dot(camera.direction, hit.normal) > 0) {
        outward_normal = -hit.normal;
        ni_over_nt = material.refractive;
        cosine = material.refractive * dot(camera.direction, hit.normal) / length(camera.direction);
    } else {
        outward_normal = hit.normal;
        ni_over_nt = 1.0f / material.refractive;
        cosine = -dot(camera.direction, hit.normal) / length(camera.direction);
    }

    // maybe refracted (who knows)
    vec3 refracted = refract(camera.direction, outward_normal, ni_over_nt);

    float reflect_prob = 1;
    if (dot(refracted, hit.normal) < 0) {
        reflect_prob = Lick(cosine, material.refractive);
    }

    if (RandomFloat(local_random, random_id) < reflect_prob) {
        camera.direction = reflect(camera.direction, hit.normal);
    } else {
        camera.direction = refracted;
    }
}

__device__ vec3 Render(const CUDA::device_ptr<Sphere>& spheres, Camera& camera, const LocalRandom& local_random, int& random_id) {

    bool is_background = true;
    vec3 current_color = vec3{1,1,1};
    float k = 2;

    for (int ray_id = 0; ray_id < 10; ray_id++) {

        // register hit
        Hit hit{-1};
        Sphere current_sphere;
        FindSphere(spheres, camera, hit, current_sphere);

        const Material& material = current_sphere.material;
        if (hit.distance > 0) {

            is_background = false;
            camera.origin = hit.position;

            if (material.diel) {
                RenderDielectric(camera, hit, material, local_random, random_id);
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

__device__ LocalRandom CreateRandomGenerator(const CUDA::device_ptr<float>& random_numbers, const int xi, const int yi, const int display_width) {
    float random_start_f = random_numbers.get()[xi + yi * display_width];
    int random_start_id = round(random_start_f * (float)random_numbers.size());

    return {
            &random_numbers.get()[random_start_id],
            static_cast<int>(random_numbers.size() - random_start_id)
    };
}

__device__ Camera CreateCamera(const int xi, const int yi, const LocalRandom& random, int& random_id, const int display_width, const int display_height) {
    float x = ((float)xi + RandomFloat(random, random_id)) / (float)display_width - 0.5f;
    float y = ((float)yi + RandomFloat(random, random_id)) / (float)display_height - 0.5f;
    float aspect = (float)display_width / (float)display_height;
    x *= aspect;
    return {{0,0,0}, {x,y, -1}};
}

__global__ void RayTracer::RenderScreen(
        CUDA::device_ptr<Sphere> spheres,
        CUDA::device_ptr<float> random_numbers,
        CUDA::device_ptr<vec3> display_ptr,
        const int display_width,
        const int display_height) {

    int xi = threadIdx.x + blockDim.x * blockIdx.x;
    int yi = threadIdx.y + blockDim.y * blockIdx.y;

    if (xi < display_width && yi < display_height) {

        LocalRandom random = CreateRandomGenerator(random_numbers, xi, yi, display_width);
        int random_id = 0;

        vec3 color{0,0,0};
        for (int i = 0; i < RAY_COUNT; i++) {
            Camera camera = CreateCamera(xi, yi, random, random_id, display_width, display_height);
            color += Render(spheres, camera, random, random_id);
        }
        color = color / (float)RAY_COUNT;
        color = {sqrtf(color.x), sqrtf(color.y), sqrtf(color.z)};

        display_ptr.get()[xi + yi * display_width] = color;
    }
}
