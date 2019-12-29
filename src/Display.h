//
// Created by Vadim Gush on 27.12.2019.
//

#ifndef RAYTRACING_DISPLAY_H
#define RAYTRACING_DISPLAY_H

#include <iostream>
#include <vector>
#include <glm/vec3.hpp>
#include "utils/cuda_unique_ptr.h"

class Display {
public:
    Display(int, int);

    cuda_device_ptr<glm::vec3> GetDisplay();

    friend std::ostream& operator<<(std::ostream&, Display& display);

private:
    cuda_unique_ptr<glm::vec3> display_;
    int width_;
    int height_;
};


#endif //RAYTRACING_DISPLAY_H
