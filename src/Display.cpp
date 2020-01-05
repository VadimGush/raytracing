//
// Created by Vadim Gush on 27.12.2019.
//

#include "Display.h"

#include <vector>

using namespace glm;
using namespace std;

Display::Display(const int width, const int height) : width_(width), height_(height), display_(width * height) {}

CUDA::device_ptr<vec3> Display::GetDisplay() {
    return display_.get_device_pointer();
}

vector<vec3> Display::GetImage() const {
    return move(display_.copy_to_vector());
}

