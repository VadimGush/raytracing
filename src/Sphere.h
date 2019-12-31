//
// Created by tsukuto on 28.12.19.
//

#ifndef RAYTRACING_SPHERE_H
#define RAYTRACING_SPHERE_H

#include <glm/vec3.hpp>

struct Material {
   glm::vec3 color;

   bool light = false;

   // amount of reflected rays by metal material
   bool metal = false;
   float reflect = 1;

   // refractive index for dielectrics
   bool diel = false;
   float refractive = 1;

   static Material Metal(const glm::vec3& c, const float r) {
       return {c, false, true, r};
   }

   static Material Scatter(const glm::vec3& c) {
       return {c};
   }

   static Material Light(const glm::vec3& c) {
       return {c, true};
   }

   static Material Dielectric(const float refractive_index) {
       return {{1,1,1}, false, false, 0, true, refractive_index};
   }

   static Material Air() {
       return {{1,1,1}, false, false, 0, true};
   }
};

struct Sphere {
    float radius;
    glm::vec3 position;
    Material material;
};

#endif //RAYTRACING_SPHERE_H
