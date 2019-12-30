//
// Created by tsukuto on 28.12.19.
//

#ifndef RAYTRACING_SPHERE_H
#define RAYTRACING_SPHERE_H

#include <glm/vec3.hpp>

struct Sphere {
    float radius;
    glm::vec3 position;
    glm::vec3 color;
    bool light = 0;
};

#endif //RAYTRACING_SPHERE_H
