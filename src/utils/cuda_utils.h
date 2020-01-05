//
// Created by Vadim Gush on 29.12.19.
//

#ifndef RAYTRACING_CUDAPOINTER_H
#define RAYTRACING_CUDAPOINTER_H

#include <cuda_runtime_api.h>
#include <exception>
#include <vector>
#include "logger.h"

namespace CUDA {

    template <typename T>
    class device_ptr {
    public:

        explicit device_ptr(T* pointer, const size_t size)
            : device_pointer_(pointer), size_(size) { }

        __host__ __device__ T* get() const {
            return device_pointer_;
        }

        __host__ __device__ size_t size() const {
            return size_;
        }

    private:
        const size_t size_;
        T* device_pointer_;
    };

    template<typename T>
    class unique_ptr {
    public:
        unique_ptr() : size_(1) {
            allocate(1);
        }

        explicit unique_ptr(const size_t size) : size_(size) {
            allocate(size);
        }

        unique_ptr(const unique_ptr &) = delete;

        unique_ptr(unique_ptr &&ptr) noexcept : device_pointer_(ptr.device_pointer_), size_(ptr.size_) {
            ptr.device_pointer_ = nullptr;
            ptr.size_ = 0;
        }

        unique_ptr &operator=(const unique_ptr &) = delete;

        unique_ptr &operator=(unique_ptr &&ptr) noexcept {
            device_pointer_ = ptr.device_pointer_;
            size_ = ptr.size_;
            ptr.device_pointer_ = nullptr;
            ptr.size_ = 0;
            return *this;
        }

        void copy_from(const T *pointer) {
            auto status = cudaMemcpy(device_pointer_, pointer, sizeof(T) * size_, cudaMemcpyHostToDevice);
            if (status)
                throw std::runtime_error("Failed to copy memory from host to device.");
        }

        void copy_to(T *pointer) const {
            auto status = cudaMemcpy(pointer, device_pointer_, sizeof(T) * size_, cudaMemcpyDeviceToHost);
            if (status)
                throw std::runtime_error("Failed to copy memory from device to host.");
        }

        std::vector<T> copy_to_vector() const {
            std::vector<T> result(size_);
            copy_to(result.data());
            return std::move(result);
        }

        device_ptr<T> get_device_pointer() const {
            return device_ptr<T>(device_pointer_, size_);
        }

        ~unique_ptr() {
            auto status = cudaFree(device_pointer_);
            if (status != cudaSuccess)
                Logger::error() << "Failed to free memory on CUDA device." << std::endl;
        }

    private:
        void allocate(const size_t size) {
            auto status = cudaMalloc(reinterpret_cast<void **>(&device_pointer_), sizeof(T) * size);
            if (status != cudaSuccess)
                throw std::runtime_error("Failed to allocate memory on CUDA device.");
        }

        T *device_pointer_;
        size_t size_;
    };

}


#endif //RAYTRACING_CUDAPOINTER_H
