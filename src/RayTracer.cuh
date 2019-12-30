//
// Created by Vadim Gush on 27.12.2019.
//

#ifndef RAYTRACING_RAYTRACER_H
#define RAYTRACING_RAYTRACER_H

#include <glm/vec3.hpp>
#include "utils/cuda_unique_ptr.h"
#include "Sphere.h"

namespace RayTracer {

    __global__ void RenderScreen(
            CUDA::device_ptr<Sphere>, const int spheres_count,
            CUDA::device_ptr<float>, const int numbers_per_thread,
            CUDA::device_ptr<glm::vec3>, const int display_width, const int display_height
    );

};


#endif //RAYTRACING_RAYTRACER_H
