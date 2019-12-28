//
// Created by Vadim Gush on 27.12.2019.
//

#include "Display.h"

#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <exception>

using namespace glm;
using namespace std;

Display::Display(const int width, const int height) : width_(width), height_(height) {
    cout << "CUDA: Allocate memory for display" << endl;
    int status = cudaMalloc(reinterpret_cast<void **>(&display_), sizeof(vec3) * width * height);

    if (status != 0)
        throw runtime_error("CUDA memory allocation failed! Try to run program again.");
}

vec3* Display::GetDisplay() {
    return display_;
}

Display::~Display() {
    cout << "CUDA: Free display memory" << endl;
    cudaFree(display_);
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

ostream& operator<<(ostream& output, const Display& d) {
    vector<vec3> result;
    result.reserve(d.width_ * d.height_);
    cudaMemcpy(result.data(), d.display_, sizeof(vec3) * d.width_ * d.height_, cudaMemcpyDeviceToHost);
    output << "P3" << "\n" << d.width_ << " " << d.height_ << "\n" << 255 << "\n";

    int index = 0;
    for (int y = d.height_ - 1; y >= 0; y--) {
        for (int x = 0; x < d.width_; x++) {
            index = d.width_ * y + x;
            vec3& pixel = result[index];
            pixel = {sqrt(pixel.r), sqrt(pixel.g), sqrt(pixel.b)};
            Filter(pixel);
            output << (int)(pixel.r * 255.9) << " " << (int)(pixel.g * 255.9) << " " << (int)(pixel.b * 255.9) << "\n";
        }
    }
    return output;
}
