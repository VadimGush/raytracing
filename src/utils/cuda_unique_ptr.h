//
// Created by tsukuto on 29.12.19.
//

#ifndef RAYTRACING_CUDAPOINTER_H
#define RAYTRACING_CUDAPOINTER_H

#include <iostream>
#include <cuda_runtime_api.h>
#include <exception>

template <typename T>
class cuda_device_ptr {
public:

    explicit cuda_device_ptr(T* pointer, const size_t size) : device_pointer_(pointer), size_(size) {}

    // host is not allowed to call __device__ functions
    // this is why this pointer can be accessible only on GPU
    __device__ T* get() {
        return device_pointer_;
    }

    // if this pointer to array then we can provide size of array
    __device__ size_t size() {
        return size_;
    }

private:
    const size_t size_;
    T* device_pointer_;
};

template <typename T>
class cuda_unique_ptr {
public:
    cuda_unique_ptr() : size_(1) {
        allocate(1);
    }

    explicit cuda_unique_ptr(const size_t size) : size_(size){
        allocate(size);
    }

    cuda_unique_ptr(const cuda_unique_ptr&) =delete;

    cuda_unique_ptr(cuda_unique_ptr&& ptr) noexcept {
        device_pointer_ = ptr.device_pointer_;
        ptr.device_pointer_ = nullptr;
    }

    cuda_unique_ptr& operator=(const cuda_unique_ptr&) =delete;

    cuda_unique_ptr& operator=(cuda_unique_ptr&& ptr) noexcept {
        device_pointer_ = ptr.device_pointer_;
        ptr.device_pointer_ = nullptr;
        return *this;
    }

    void copy_from(const T* pointer) {
        auto status = cudaMemcpy(device_pointer_, pointer, sizeof(T) * size_, cudaMemcpyHostToDevice);
        if (status)
            throw std::runtime_error("Failed to copy memory from host to device.");
    }

    void copy_to(T* pointer) {
        auto status = cudaMemcpy(pointer, device_pointer_, sizeof(T) * size_, cudaMemcpyDeviceToHost);
        if (status)
            throw std::runtime_error("Failed to copy memory from device to host.");
    }

    cuda_device_ptr<T> get_device_pointer() {
        return cuda_device_ptr<T>(device_pointer_, size_);
    }

    ~cuda_unique_ptr() {
        auto status = cudaFree(device_pointer_);
        if (status != cudaSuccess)
            std::cout << "ERROR: Failed to free memory on CUDA device.";
    }
private:
    void allocate(const size_t size) {
        auto status = cudaMalloc(reinterpret_cast<void**>(&device_pointer_), sizeof(T) * size);
        if (status != cudaSuccess)
            throw std::runtime_error("Failed to allocate memory on CUDA device.");
    }

    T* device_pointer_;
    const size_t size_;
};


#endif //RAYTRACING_CUDAPOINTER_H
