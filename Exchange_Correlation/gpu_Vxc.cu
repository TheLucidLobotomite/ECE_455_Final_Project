/**
 * Exchange-Correlation Potential (CUDA GPU - OPTIMIZED)
 * Optimizations for iterative usage:
 * - Persistent GPU memory allocation
 * - Pinned host memory for faster transfers
 * - Context-based to avoid repeated malloc/free
 */

#ifndef GPU_VXC_OPTIMIZED_CU_H
#define GPU_VXC_OPTIMIZED_CU_H

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

// Parameters for epsilon_c(r_s, 0) from T-Money's Table I
__constant__ double d_A = 0.031091;
__constant__ double d_alpha1 = 0.21370;
__constant__ double d_beta1 = 7.5957;
__constant__ double d_beta2 = 3.5876;
__constant__ double d_beta3 = 1.6382;
__constant__ double d_beta4 = 0.49294;

/**
 * LDA Exchange Potential
 */
__device__ inline double vx_lda_gpu(double n) {
    const double Cx = 0.738558766382022;
    return -Cx * pow(n, 1.0/3.0);
}

/**
 * Correlation energy per electron
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
 * LDA Correlation Potential
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
 * Context structure for persistent GPU memory
 */
struct VxcContext {
    double *d_n;           // Device memory for input density
    double *d_vxc;         // Device memory for output potential
    double *h_n_pinned;    // Pinned host memory for input (optional)
    double *h_vxc_pinned;  // Pinned host memory for output (optional)
    size_t capacity;       // Maximum size allocated
    bool use_pinned;       // Whether pinned memory is allocated
};

/**
 * Initialize context with persistent GPU allocations
 * 
 * @param max_size Maximum number of grid points expected
 * @param use_pinned_memory If true, also allocates pinned host memory for faster transfers
 * @return Pointer to initialized context
 */
inline VxcContext* vxc_init(size_t max_size, bool use_pinned_memory = true) {
    VxcContext* ctx = new VxcContext;
    ctx->capacity = max_size;
    ctx->use_pinned = use_pinned_memory;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&ctx->d_n, max_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx->d_vxc, max_size * sizeof(double)));
    
    // Optionally allocate pinned host memory for faster PCIe transfers
    if (use_pinned_memory) {
        CUDA_CHECK(cudaMallocHost(&ctx->h_n_pinned, max_size * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&ctx->h_vxc_pinned, max_size * sizeof(double)));
    } else {
        ctx->h_n_pinned = nullptr;
        ctx->h_vxc_pinned = nullptr;
    }
    
    return ctx;
}

/**
 * Compute Vxc using persistent context
 * 
 * @param ctx Context with pre-allocated memory
 * @param h_n Host array with electron density (can be regular or pinned)
 * @param h_vxc Host array for output potential (can be regular or pinned)
 * @param size Number of grid points (must be <= ctx->capacity)
 */
inline void vxc_compute(VxcContext* ctx, const double* h_n, double* h_vxc, size_t size) {
    if (size > ctx->capacity) {
        fprintf(stderr, "Error: size %zu exceeds allocated capacity %zu\n", size, ctx->capacity);
        exit(EXIT_FAILURE);
    }
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(ctx->d_n, h_n, size * sizeof(double), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    vxc_kernel<<<numBlocks, blockSize>>>(ctx->d_n, ctx->d_vxc, size);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_vxc, ctx->d_vxc, size * sizeof(double), cudaMemcpyDeviceToHost));
}

/**
 * Compute Vxc using pinned memory buffers (fastest option)
 * Only works if context was initialized with use_pinned_memory=true
 * 
 * Usage pattern:
 *   1. Copy your data to ctx->h_n_pinned
 *   2. Call vxc_compute_pinned()
 *   3. Read results from ctx->h_vxc_pinned
 * 
 * @param ctx Context with pinned memory allocated
 * @param size Number of grid points
 */
inline void vxc_compute_pinned(VxcContext* ctx, size_t size) {
    if (!ctx->use_pinned) {
        fprintf(stderr, "Error: context not initialized with pinned memory\n");
        exit(EXIT_FAILURE);
    }
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
inline void vxc_cleanup(VxcContext* ctx) {
    if (ctx) {
        CUDA_CHECK(cudaFree(ctx->d_n));
        CUDA_CHECK(cudaFree(ctx->d_vxc));
        
        if (ctx->use_pinned) {
            CUDA_CHECK(cudaFreeHost(ctx->h_n_pinned));
            CUDA_CHECK(cudaFreeHost(ctx->h_vxc_pinned));
        }
        
        delete ctx;
    }
}

// ============================================================================
// Legacy API (for backward compatibility - not recommended for iterative use)
// ============================================================================

/**
 * One-shot calculation (legacy interface)
 * NOTE: For iterative calculations, use the context API above instead!
 */
inline void calculate_vxc_cuda(const double* h_n, double* h_vxc, size_t size) {
    VxcContext* ctx = vxc_init(size, false);
    vxc_compute(ctx, h_n, h_vxc, size);
    vxc_cleanup(ctx);
}

#endif // GPU_VXC_OPTIMIZED_CU_H