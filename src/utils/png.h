//
// Created by Vadim Gush on 05.01.2020.
//

#ifndef RAYTRACING_PNG_H
#define RAYTRACING_PNG_H

#include <string>
#include <glm/vec3.hpp>
#include <vector>

namespace PNG {

    void WriteImage(const std::string& path, int width, int height, const std::vector<glm::vec3>& image);

}

#endif //RAYTRACING_PNG_H
