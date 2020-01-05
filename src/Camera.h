//
// Created by Vadim Gush on 03.01.20.
//

#ifndef RAYTRACING_CAMERA_H
#define RAYTRACING_CAMERA_H

#include <glm/vec3.hpp>

struct Camera {
    glm::vec3 origin;
    glm::vec3 direction;
};

#endif //RAYTRACING_CAMERA_H
