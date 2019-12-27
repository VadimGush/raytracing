//
// Created by Vadim Gush on 27.12.2019.
//

#ifndef RAYTRACING_DISPLAY_H
#define RAYTRACING_DISPLAY_H

#include <iostream>
#include <vector>
#include <glm/vec3.hpp>

class Display {
public:
    Display(int, int);

    glm::vec3* GetDisplay();

    ~Display();

    friend std::ostream& operator<<(std::ostream&, const Display& display);

private:
    glm::vec3* display_;
    int width_;
    int height_;
};


#endif //RAYTRACING_DISPLAY_H
