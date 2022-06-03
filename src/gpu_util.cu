#include <stdio.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include "gpu_util.h"

//#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#define ALIGNED(x,y) (((x)%(y)) ? (x)+(y)-((x)%(y)) : x)
#define checkcuerr() \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error %s at %s:%d\n", \
                cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


#define checkcublaserr(stat) \
    do { \
        if (stat != CUBLAS_STATUS_SUCCESS) { \
            fprintf (stderr, "CUBLAS failed: %s:%d\n", __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)



#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define CLIP(a,b,c) (((a)<(b))?(b):((a)>(c)?(c):(a)))


namespace util::gpu {

//cublasHandle_t HANDLE;
cublasHandle_t HANDLES[2] = {0};


size_t get_device_handle(int device) {
    return reinterpret_cast<size_t>(HANDLES[device]);
}


void cu_cublasCreate_once(int device)
{
    if (device < 2 && !HANDLES[device]) {
        int cur_device;
        cudaGetDevice(&cur_device);
        cublasStatus_t stat;
        cudaSetDevice(device);
        stat = cublasCreate(&HANDLES[device]);
        checkcublaserr(stat);
        cudaSetDevice(cur_device);
    }
//    cublasStatus_t stat = cublasCreate(&HANDLE);
//    checkcublaserr(stat);
}


void cu_cudaDeviceSynchronize()
{
    cudaDeviceSynchronize();
    checkcuerr();
}


void cu_cudaFree(void* ptr)
{
    cudaFree(ptr);
    checkcuerr();
}


void cu_cudaFreeHost(void* ptr)
{
    cudaFreeHost(ptr);
    checkcuerr();
}

void cu_cudaMallocHost(void** ptr, unsigned size)
{
    cudaMallocHost(ptr, size);
    checkcuerr();
}

void cu_cudaMalloc(void** ptr, unsigned size)
{
    cudaMalloc(ptr, size);
    checkcuerr();
    cudaDeviceSynchronize();
    checkcuerr();
}


void cu_cudaMemcpy_HD(void* dst, const void* src, size_t count)
{
    cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
    checkcuerr();
}

void cu_cudaMemcpy_DH(void* dst, const void* src, size_t count)
{
    cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
    checkcuerr();
}

void cu_cudaMemcpy_DD(void* dst, const void* src, size_t count)
{
    cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
    checkcuerr();
}

void cu_cudaMemGetInfo(size_t* free, size_t* total)
{
    cudaMemGetInfo(free, total);
}

void cu_cudaSetDevice(int device)
{
    cudaSetDevice(device);
    cudaDeviceSynchronize();
    checkcuerr();
}

int cu_cudaGetDevice()
{
    int device;
    cudaGetDevice(&device);
    return device;
}

} // util::gpu
