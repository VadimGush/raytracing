//
// Created by Vadim Gush on 05.01.2020.
//

#include "png.h"

#include "c_file.h"
#include <iostream>
#include <png.h>

using namespace std;

void WriteImage(const png_structp& png_ptr, const int width, const int height, const vector<glm::vec3>& image) {
    for (int y = height - 1; y >= 0; --y) {
        auto* row = new png_byte[width * 3];
        for (int x = 0; x < width; ++x) {
            int index = x + y * width;
            const glm::vec3& pixel = image[index];
            row[x * 3 + 0] = (unsigned char)(pixel.r * 255.9);
            row[x * 3 + 1] = (unsigned char)(pixel.g * 255.9);
            row[x * 3 + 2] = (unsigned char)(pixel.b * 255.9);
        }
        png_write_row(png_ptr, row);
        delete[] row;
    }
    png_write_end(png_ptr, nullptr);
}

void PNG::WriteImage(const string& path, const int width, const int height, const vector<glm::vec3>& image) {

    c_file file(path, "wb");
    if (!file.is_open()) {
        throw runtime_error("Cannot create image file");
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) {
        throw runtime_error("Cannot create image structure");
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, nullptr);
        throw runtime_error("Cannot create image info");
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        throw runtime_error("Cannot initialize IO");
    }

    png_init_io(png_ptr, file.get_ptr());

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        throw runtime_error("Cannot write image info");
    }

    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB,
            PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        throw runtime_error("Cannot write image");
    }

    WriteImage(png_ptr, width, height, image);
}

