//
// Created by Vadim Gush on 27.12.2019.
//

#ifndef RAYTRACING_RAYTRACER_H
#define RAYTRACING_RAYTRACER_H

#include <glm/vec3.hpp>
#include "utils/cuda_memory.h"
#include "Sphere.h"

namespace RayTracer {

    __global__ void RenderScreen(
            CUDA::device_ptr<Sphere>,
            CUDA::device_ptr<float>,
            CUDA::device_ptr<glm::vec3>, int display_width, int display_height
    );

};


#endif //RAYTRACING_RAYTRACER_H
