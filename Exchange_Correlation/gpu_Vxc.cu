/**
 * Exchange-Correlation Potential - Perdew-Wang 1992 (CUDA GPU - OPTIMIZED)
 * Optimized for iterative usage with persistent allocation and pinned memory
 * 
 * Reference: Phys. Rev. B 45, 13244 (1992)
 * Units: RYDBERG atomic units (to match CPU implementation)
 */

#ifndef GPU_VXC_CU_H
#define GPU_VXC_CU_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

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
// CONVERTED TO RYDBERG UNITS (original Hartree values × 2)
__constant__ double d_A = 0.062182;        // was 0.031091 in Hartree
__constant__ double d_alpha1 = 0.21370;    // dimensionless
__constant__ double d_beta1 = 7.5957;      // dimensionless
__constant__ double d_beta2 = 3.5876;      // dimensionless
__constant__ double d_beta3 = 1.6382;      // dimensionless
__constant__ double d_beta4 = 0.49294;     // dimensionless

/**
 * LDA Exchange Potential - GPU version (Rydberg units)
 */
__device__ inline double vx_lda_gpu(double n) {
    const double Cx = 1.477117532764044;  // Rydberg units
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
 * Context structure for persistent GPU memory with pinned host memory
 */
struct VxcContext {
    double *d_n;           // Device memory for input density
    double *d_vxc;         // Device memory for output potential
    double *h_n_pinned;    // Pinned host memory for input
    double *h_vxc_pinned;  // Pinned host memory for output
    size_t capacity;       // Maximum size allocated
};

/**
 * Initialize context with persistent GPU allocations and pinned host memory
 * 
 * @param max_size Maximum number of grid points expected
 * @return Pointer to initialized context
 */
extern "C" VxcContext* vxc_init(size_t max_size) {
    VxcContext* ctx = new VxcContext;
    ctx->capacity = max_size;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&ctx->d_n, max_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx->d_vxc, max_size * sizeof(double)));
    
    // Allocate pinned host memory for faster PCIe transfers
    CUDA_CHECK(cudaMallocHost(&ctx->h_n_pinned, max_size * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&ctx->h_vxc_pinned, max_size * sizeof(double)));
    
    return ctx;
}

/**
 * Compute Vxc using pinned memory buffers (fastest option)
 * 
 * Usage pattern:
 *   1. Copy your data to ctx->h_n_pinned
 *   2. Call vxc_compute_pinned()
 *   3. Read results from ctx->h_vxc_pinned
 * 
 * @param ctx Context with pinned memory allocated
 * @param size Number of grid points
 */
extern "C" void vxc_compute_pinned(VxcContext* ctx, size_t size) {
    if (size > ctx->capacity) {
        fprintf(stderr, "Error: size %zu exceeds allocated capacity %zu\n", size, ctx->capacity);
        exit(EXIT_FAILURE);
    }
    
    // Copy from pinned host to device
    CUDA_CHECK(cudaMemcpy(ctx->d_n, ctx->h_n_pinned, size * sizeof(double), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    vxc_kernel<<<numBlocks, blockSize>>>(ctx->d_n, ctx->d_vxc, size);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy from device to pinned host
    CUDA_CHECK(cudaMemcpy(ctx->h_vxc_pinned, ctx->d_vxc, size * sizeof(double), cudaMemcpyDeviceToHost));
}

/**
 * Cleanup context and free all allocated memory
 */
extern "C" void vxc_cleanup(VxcContext* ctx) {
    if (ctx) {
        CUDA_CHECK(cudaFree(ctx->d_n));
        CUDA_CHECK(cudaFree(ctx->d_vxc));
        CUDA_CHECK(cudaFreeHost(ctx->h_n_pinned));
        CUDA_CHECK(cudaFreeHost(ctx->h_vxc_pinned));
        delete ctx;
    }
}

#endif // GPU_VXC_CU_H