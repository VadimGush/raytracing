//
// Created by Vadim Gush on 27.12.2019.
//

#ifndef RAYTRACING_DISPLAY_H
#define RAYTRACING_DISPLAY_H

#include <iostream>
#include <vector>
#include <glm/vec3.hpp>
#include "utils/cuda_utils.h"

class Display {
public:
    Display(int, int);

    CUDA::device_ptr<glm::vec3> GetDisplay();

    std::vector<glm::vec3> GetImage() const;
private:
    CUDA::unique_ptr<glm::vec3> display_;
    int width_;
    int height_;
};


#endif //RAYTRACING_DISPLAY_H
