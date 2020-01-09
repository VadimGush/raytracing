//
// Created by Vadim Gush on 05.01.2020.
//

#include "file.h"

using namespace C;

file::file(const std::string& path, const std::string& mode) {
    file_ptr_ = fopen(path.c_str(), mode.c_str());
}

file::file(file&& file) noexcept {
    file_ptr_ = file.file_ptr_;
    file.file_ptr_ = nullptr;
}

file& file::operator=(file&& file) noexcept {
    file_ptr_ = file.file_ptr_;
    file.file_ptr_ = nullptr;
    return *this;
}

bool file::is_open() const {
    return file_ptr_;
}

FILE* file::get_ptr() {
    return file_ptr_;
}

file::~file() {
    fclose(file_ptr_);
}



