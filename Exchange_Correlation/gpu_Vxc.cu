/**
 * Exchange-Correlation Potential - Perdew-Wang 1992 (CUDA GPU)
 * Simplified implementation: Vxc = Vx + Vc (unpolarized, zeta=0)
 * 
 * Reference: Phys. Rev. B 45, 13244 (1992)
 */

#ifndef GPU_VXC_CU_H
#define GPU_VXC_CU_H

#include <cuda_runtime.h>

#define PI_CUDA 3.14159265358979323846

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Parameters for epsilon_c(r_s, 0) from Table I - unpolarized only
__constant__ double d_A = 0.031091;
__constant__ double d_alpha1 = 0.21370;
__constant__ double d_beta1 = 7.5957;
__constant__ double d_beta2 = 3.5876;
__constant__ double d_beta3 = 1.6382;
__constant__ double d_beta4 = 0.49294;

/**
 * LDA Exchange Potential - GPU version
 */
__device__ inline double vx_lda_gpu(double n) {
    const double Cx = 0.738558766382022;
    return -Cx * pow(n, 1.0/3.0);
}

/**
 * Correlation energy per electron - GPU version
 */
__device__ inline double epsilon_c_gpu(double rs) {
    double rs_sqrt = sqrt(rs);
    double rs_3_2 = rs * rs_sqrt;
    double rs_2 = rs * rs;
    
    double Q1 = 2.0 * d_A * (d_beta1 * rs_sqrt + d_beta2 * rs + 
                              d_beta3 * rs_3_2 + d_beta4 * rs_2);
    
    return -2.0 * d_A * (1.0 + d_alpha1 * rs) * log(1.0 + 1.0 / Q1);
}

/**
 * LDA Correlation Potential - GPU version
 */
__device__ inline double vc_pw92_gpu(double n) {
    const double rs = pow(3.0 / (4.0 * PI_CUDA * n), 1.0/3.0);
    
    const double h = 1e-6;
    double ec = epsilon_c_gpu(rs);
    double ec_plus = epsilon_c_gpu(rs + h);
    double d_ec_drs = (ec_plus - ec) / h;
    
    return ec - (rs / 3.0) * d_ec_drs;
}

/**
 * CUDA kernel for calculating Vxc
 */
__global__ void vxc_kernel(const double* n, double* vxc, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        double n_safe = fmax(n[idx], 1e-12);
        double Vx = vx_lda_gpu(n_safe);
        double Vc = vc_pw92_gpu(n_safe);
        vxc[idx] = Vx + Vc;
    }
}

/**
 * Host function to calculate Vxc using CUDA
 */
inline void calculate_vxc_cuda(const double* h_n, double* h_vxc, size_t size) {
    // Allocate device memory
    double *d_n, *d_vxc;
    CUDA_CHECK(cudaMalloc(&d_n, size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vxc, size * sizeof(double)));
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_n, h_n, size * sizeof(double), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    vxc_kernel<<<numBlocks, blockSize>>>(d_n, d_vxc, size);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_vxc, d_vxc, size * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_n));
    CUDA_CHECK(cudaFree(d_vxc));
}

#endif // GPU_VXC_CU_H