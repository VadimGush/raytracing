//
// Created by Vadim Gush on 27.12.2019.
//

#ifndef RAYTRACING_RAYTRACER_H
#define RAYTRACING_RAYTRACER_H

#include <glm/vec3.hpp>
#include "Sphere.h"

namespace RayTracer {

    __global__ void RenderScreen(Sphere* spheres, const int spheres_count, glm::vec3* display, const int display_width, const int display_height);

};


#endif //RAYTRACING_RAYTRACER_H
