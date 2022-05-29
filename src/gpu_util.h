#pragma once

namespace util::gpu {
    
size_t get_device_handle(int device);
void cu_cudaFree(void* ptr);
void cu_cudaFreeHost(void* ptr);
void cu_cudaMallocHost(void** ptr, unsigned size);
void cu_cudaMalloc(void** ptr, unsigned size);
void cu_cudaMemcpy_HD(void* dst, const void* src, size_t count);
void cu_cudaMemcpy_DH(void* dst, const void* src, size_t count);
void cu_cudaMemcpy_DD(void* dst, const void* src, size_t count);
void cu_cudaDeviceSynchronize();
void cu_cublasCreate_once(int device);
void cu_cudaMemGetInfo(size_t* free, size_t* total);
void cu_cudaSetDevice(int device);
int cu_cudaGetDevice();
    
}
