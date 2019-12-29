//
// Created by Vadim Gush on 27.12.2019.
//

#include "Display.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <exception>

using namespace glm;
using namespace std;

Display::Display(const int width, const int height) : width_(width), height_(height), display_(width * height) {}

vec3* Display::GetDisplay() {
    return display_.get_pointer();
}

inline void Clamp(float& value) {
    if (value < 0) value = 0;
    if (value > 1) value = 1;
}

inline void Filter(vec3& v) {
    Clamp(v.x);
    Clamp(v.y);
    Clamp(v.z);
}

ostream& operator<<(ostream& output, Display& d) {
    vector<vec3> result;
    result.reserve(d.width_ * d.height_);

    d.display_.copy_to(result.data());

    output << "P3" << "\n" << d.width_ << " " << d.height_ << "\n" << 255 << "\n";

    int index = 0;
    for (int y = d.height_ - 1; y >= 0; y--) {
        for (int x = 0; x < d.width_; x++) {
            index = d.width_ * y + x;
            vec3& pixel = result[index];
            Filter(pixel);
            pixel = {sqrt(pixel.r), sqrt(pixel.g), sqrt(pixel.b)};
            output << (int)(pixel.r * 255.9) << " " << (int)(pixel.g * 255.9) << " " << (int)(pixel.b * 255.9) << "\n";
        }
    }
    return output;
}
