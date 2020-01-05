//
// Created by Vadim Gush on 05.01.2020.
//

#ifndef RAYTRACING_FILE_H
#define RAYTRACING_FILE_H

#include <string>

namespace C {
    class file {
    public:
        file(const std::string &path, const std::string &mode);

        file(const file &) = delete;

        file &operator=(const file &) = delete;

        file(file &&) noexcept;

        file &operator=(file &&) noexcept;

        bool is_open() const;

        FILE *get_ptr();

        ~file();

    private:
        FILE *file_ptr_ = nullptr;
    };
}


#endif //RAYTRACING_FILE_H
