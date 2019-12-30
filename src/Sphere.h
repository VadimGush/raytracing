//
// Created by tsukuto on 28.12.19.
//

#ifndef RAYTRACING_SPHERE_H
#define RAYTRACING_SPHERE_H

#include <glm/vec3.hpp>

struct Material {
   glm::vec3 color;
   bool light = false;
   bool metal = false;
   float reflect = 1;

   static Material Metal(const glm::vec3& c, const float r) {
       return {c, false, true, r};
   }

   static Material Scatter(const glm::vec3& c) {
       return {c};
   }

   static Material Light(const glm::vec3& c) {
       return {c, true};
   }
};

struct Sphere {
    float radius;
    glm::vec3 position;
    Material material;
};

#endif //RAYTRACING_SPHERE_H
