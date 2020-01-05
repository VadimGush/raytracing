//
// Created by Vadim Gush on 05.01.2020.
//

#ifndef RAYTRACING_C_FILE_H
#define RAYTRACING_C_FILE_H

#include <string>

class c_file {
public:
    c_file(const std::string& path, const std::string& mode);

    c_file(const c_file&) =delete;
    c_file& operator=(const c_file&) =delete;

    c_file(c_file&&) noexcept;
    c_file& operator=(c_file&&) noexcept;

    bool is_open() const;

    FILE* get_ptr();

    ~c_file();

private:
    FILE* file_ptr_ = nullptr;
};


#endif //RAYTRACING_C_FILE_H
