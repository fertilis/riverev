#include <stdio.h>
#include "cublas_v2.h"
#include "gpu_util.h"
#include "gpu_calc.h"

#define CUDA_NAN __int_as_float(0x7fffffff)

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



__global__ void
k_calc_value(
        float* values, 
        const float* gainsums, 
        const float* weightsums, 
        const float* board_compatibility, 
        const float pot, 
        const float cost, 
        const float exp)
{
    int hand_index = blockDim.x*blockIdx.x + threadIdx.x; 
    if (hand_index < 1326) {
        float weightsum = weightsums[hand_index];
        float equity;
        if (weightsum) {
            equity = gainsums[hand_index]/weightsum;
            values[hand_index] = powf(equity, exp) * pot - cost;
        } else {
            if (board_compatibility[hand_index]) {
                values[hand_index] = 0.0f;
            } else {
                values[hand_index] = CUDA_NAN;
            }
        }
    }
}


namespace riverev::gpucalc {

void
calc_node_values(
        const float* hand_compatibility,
        const float* board_compatibility,
        const float* ranking,
        const float* opponent_weights,
        const float pot,
        const float cost,
        const float exp,
        float* values,
        float* gainsums,
        float* weightsums)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t stat;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_ALGO1;
    
    int device;
    cudaGetDevice(&device);
    cublasHandle_t handle = (cublasHandle_t)util::gpu::get_device_handle(device);
        
    int m = 1326, n = 1, k = 1326;
    const void* L;
    const void* R;
    void* P;

    // R W = A (gainsums)
    L = (const void*)(ranking);
    R = (const void*)(opponent_weights);
    P = (void*)(gainsums);
    stat = cublasGemmEx(
        handle, 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        m, n, k, 
        &alpha, 
        L, CUDA_R_32F, m, 
        R, CUDA_R_32F, k, 
        &beta, 
        P, CUDA_R_32F, m,
        CUBLAS_COMPUTE_32F, algo
    );
    checkcublaserr(stat);
    cudaDeviceSynchronize();
    
    // C W = B (weightsums)
    L = (const void*)(hand_compatibility);
    R = (const void*)(opponent_weights);
    P = (void*)(weightsums);
    stat = cublasGemmEx(
        handle, 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        m, n, k, 
        &alpha, 
        L, CUDA_R_32F, m, 
        R, CUDA_R_32F, k, 
        &beta, 
        P, CUDA_R_32F, m,
        CUBLAS_COMPUTE_32F, algo
    );
    checkcublaserr(stat);
    cudaDeviceSynchronize();
    
    const int blockDimX = 256;
    dim3 dimBlock(blockDimX);
    dim3 dimGrid((1326+blockDimX-1)/blockDimX);

    k_calc_value<<<dimGrid, dimBlock>>>(
            values, 
            gainsums, 
            weightsums, 
            board_compatibility, 
            pot, 
            cost, 
            exp);
    cudaDeviceSynchronize();
    checkcuerr();
}


} // riverev::gpucalc
