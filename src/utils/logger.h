//
// Created by Vadim Gush on 04.01.2020.
//

#ifndef RAYTRACING_LOGGER_H
#define RAYTRACING_LOGGER_H

#include <iostream>
#include <string>

namespace Logger {

    std::ostream& info();

    std::ostream& error();

    std::ostream& fatal();

}

#endif //RAYTRACING_LOGGER_H
