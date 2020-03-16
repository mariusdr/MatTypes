#ifndef LIBCUMANIP_CUDA_UTILS_H
#define LIBCUMANIP_CUDA_UTILS_H

#include <iostream>

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>

#ifdef __CUDA_ARCH__
#include <cuda_runtime.h>

#define CUDART_INF_F __int_as_float(0x7f800000)


#define CUDA_ASSERT(call)                                                                         \
{                                                                                                 \
    const cudaError_t error = call;                                                               \
    if (error != cudaSuccess)                                                                     \
    {                                                                                             \
        std::cerr << "Cuda Error in " << __FILE__ << " line " << __LINE__ << "\n";                \
        std::cerr << "Error Code: " << error << " Reason: " << cudaGetErrorString(error) << "\n"; \
        exit(1);                                                                                  \
    }                                                                                             \
}

__host__ inline 
void print_version_str()
{
    int cudaRuntimeVers;
    int cudaDriverVers;
    cudaRuntimeGetVersion(&cudaRuntimeVers);
    cudaDriverGetVersion(&cudaDriverVers);
    std::cout << "Using Cuda Runtime Version " << (float(cudaRuntimeVers) / 1000) << "\n";
    std::cout << "Using Cuda Driver Version " << (float(cudaDriverVers) / 1000) << "\n";
    std::cout << "Using Thrust Version " << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << "\n";
}
#endif

__host__ __device__ inline
float infty() 
{
    #ifdef __CUDA_ARCH__
    return CUDART_INF_F;
    #else 
    return INFINITY;
    #endif
}

__host__ __device__ inline 
float _sqrtf(float x)
{
    #ifdef __CUDA__ARCH__
    return __fsqrt_rd(x);
    #else 
    return sqrtf(x);
    #endif
}


#endif