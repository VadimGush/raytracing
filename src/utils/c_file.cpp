//
// Created by Vadim Gush on 05.01.2020.
//

#include "c_file.h"

c_file::c_file(const std::string& path, const std::string& mode) {
    file_ptr_ = fopen(path.c_str(), mode.c_str());
}

c_file::c_file(c_file&& file) noexcept {
    file_ptr_ = file.file_ptr_;
    file.file_ptr_ = nullptr;
}

c_file& c_file::operator=(c_file&& file) noexcept {
    file_ptr_ = file.file_ptr_;
    file.file_ptr_ = nullptr;
    return *this;
}

bool c_file::is_open() const {
    return file_ptr_;
}

FILE* c_file::get_ptr() {
    return file_ptr_;
}

c_file::~c_file() {
    fclose(file_ptr_);
}



