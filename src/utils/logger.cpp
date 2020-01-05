//
// Created by Vadim Gush on 04.01.2020.
//
#include "logger.h"

std::ostream& Logger::info() {
    return std::cout << "INFO  -> ";
}

std::ostream& Logger::error() {
    return std::cout << "ERROR -> ";
}

std::ostream& Logger::fatal() {
    return std::cout << "FATAL -> ";
}


