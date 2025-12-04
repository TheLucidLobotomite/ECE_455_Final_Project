/**
 * Exchange-Correlation Potential - Perdew-Wang 1992 (CUDA GPU - OPTIMIZED)
 * With lattice vector support and density function evaluation
 * 
 * Reference: Phys. Rev. B 45, 13244 (1992)
 * Units: RYDBERG atomic units
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

// Lattice vectors in constant memory
__constant__ double d_a1[3];
__constant__ double d_a2[3];
__constant__ double d_a3[3];
__constant__ int d_Nx, d_Ny, d_Nz;

/**
 * Density function type - user provides this function
 * Input: pointer to position [x, y, z] in Cartesian coordinates (Bohr)
 * Output: density at that point (electrons/Bohr³)
 */
typedef double (*DensityFunction)(const double* r);

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
 * CUDA kernel for calculating Vxc with pre-computed density
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
 * CUDA kernel for computing grid positions
 * Converts (i,j,k) indices to Cartesian coordinates using lattice vectors
 */
__global__ void compute_positions_kernel(double* positions, size_t total_points) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_points) {
        // Convert linear index to (i,j,k)
        int k = idx % d_Nz;
        int j = (idx / d_Nz) % d_Ny;
        int i = idx / (d_Ny * d_Nz);
        
        // Fractional coordinates
        double fi = (double)i / (double)d_Nx;
        double fj = (double)j / (double)d_Ny;
        double fk = (double)k / (double)d_Nz;
        
        // Cartesian position: r = fi*a1 + fj*a2 + fk*a3
        positions[3*idx + 0] = fi * d_a1[0] + fj * d_a2[0] + fk * d_a3[0];
        positions[3*idx + 1] = fi * d_a1[1] + fj * d_a2[1] + fk * d_a3[1];
        positions[3*idx + 2] = fi * d_a1[2] + fj * d_a2[2] + fk * d_a3[2];
    }
}

/**
 * Context structure for persistent GPU memory with lattice information
 */
struct VxcContext {
    double *d_positions;   // Device memory for grid positions (x,y,z for each point)
    double *d_n;           // Device memory for input density
    double *d_vxc;         // Device memory for output potential
    double *h_positions;   // Host memory for positions
    double *h_n_pinned;    // Pinned host memory for density
    double *h_vxc_pinned;  // Pinned host memory for output
    size_t capacity;       // Maximum number of grid points
    int Nx, Ny, Nz;        // Grid dimensions
    double a1[3], a2[3], a3[3];  // Lattice vectors (host copy)
};

/**
 * Initialize context with lattice vectors and grid dimensions
 * 
 * @param a1 First lattice vector [x, y, z] in Bohr
 * @param a2 Second lattice vector [x, y, z] in Bohr
 * @param a3 Third lattice vector [x, y, z] in Bohr
 * @param Nx Number of grid points along a1
 * @param Ny Number of grid points along a2
 * @param Nz Number of grid points along a3
 * @return Pointer to initialized context
 */
inline VxcContext* vxc_init(const double a1[3], const double a2[3], const double a3[3],
                            int Nx, int Ny, int Nz) {
    VxcContext* ctx = new VxcContext;
    ctx->Nx = Nx;
    ctx->Ny = Ny;
    ctx->Nz = Nz;
    ctx->capacity = Nx * Ny * Nz;
    
    // Store lattice vectors
    for (int i = 0; i < 3; i++) {
        ctx->a1[i] = a1[i];
        ctx->a2[i] = a2[i];
        ctx->a3[i] = a3[i];
    }
    
    // Copy lattice vectors and dimensions to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_a1, a1, 3 * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_a2, a2, 3 * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_a3, a3, 3 * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_Nx, &Nx, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_Ny, &Ny, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_Nz, &Nz, sizeof(int)));
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&ctx->d_positions, 3 * ctx->capacity * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx->d_n, ctx->capacity * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx->d_vxc, ctx->capacity * sizeof(double)));
    
    // Allocate host memory
    ctx->h_positions = new double[3 * ctx->capacity];
    CUDA_CHECK(cudaMallocHost(&ctx->h_n_pinned, ctx->capacity * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&ctx->h_vxc_pinned, ctx->capacity * sizeof(double)));
    
    // Compute grid positions on GPU
    int blockSize = 256;
    int numBlocks = (ctx->capacity + blockSize - 1) / blockSize;
    compute_positions_kernel<<<numBlocks, blockSize>>>(ctx->d_positions, ctx->capacity);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy positions back to host for density function evaluation
    CUDA_CHECK(cudaMemcpy(ctx->h_positions, ctx->d_positions, 
                         3 * ctx->capacity * sizeof(double), cudaMemcpyDeviceToHost));
    
    return ctx;
}

/**
 * Compute densities at all grid points using user-provided function
 * This runs on CPU since we can't call arbitrary host functions from GPU
 * 
 * @param ctx Context with grid positions
 * @param density_func User-provided function that returns density at a point
 */
inline void vxc_compute_density(VxcContext* ctx, DensityFunction density_func) {
    for (size_t i = 0; i < ctx->capacity; i++) {
        double* r = &ctx->h_positions[3 * i];
        ctx->h_n_pinned[i] = density_func(r);
    }
}

/**
 * Compute Vxc after density has been evaluated
 * 
 * @param ctx Context with density already computed in h_n_pinned
 */
inline void vxc_compute(VxcContext* ctx) {
    // Copy density from pinned host to device
    CUDA_CHECK(cudaMemcpy(ctx->d_n, ctx->h_n_pinned, 
                         ctx->capacity * sizeof(double), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (ctx->capacity + blockSize - 1) / blockSize;
    vxc_kernel<<<numBlocks, blockSize>>>(ctx->d_n, ctx->d_vxc, ctx->capacity);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result from device to pinned host
    CUDA_CHECK(cudaMemcpy(ctx->h_vxc_pinned, ctx->d_vxc, 
                         ctx->capacity * sizeof(double), cudaMemcpyDeviceToHost));
}

/**
 * Complete workflow: evaluate density and compute Vxc
 * 
 * @param ctx Context initialized with lattice vectors
 * @param density_func User-provided density function
 */
inline void vxc_compute_full(VxcContext* ctx, DensityFunction density_func) {
    vxc_compute_density(ctx, density_func);
    vxc_compute(ctx);
}

/**
 * Get the (i,j,k) grid index from linear index
 */
inline void vxc_get_indices(VxcContext* ctx, size_t linear_idx, int* i, int* j, int* k) {
    *k = linear_idx % ctx->Nz;
    *j = (linear_idx / ctx->Nz) % ctx->Ny;
    *i = linear_idx / (ctx->Ny * ctx->Nz);
}

/**
 * Get linear index from (i,j,k) grid indices
 */
inline size_t vxc_get_linear_index(VxcContext* ctx, int i, int j, int k) {
    return i * (ctx->Ny * ctx->Nz) + j * ctx->Nz + k;
}

/**
 * Cleanup context and free all allocated memory
 */
inline void vxc_cleanup(VxcContext* ctx) {
    if (ctx) {
        CUDA_CHECK(cudaFree(ctx->d_positions));
        CUDA_CHECK(cudaFree(ctx->d_n));
        CUDA_CHECK(cudaFree(ctx->d_vxc));
        delete[] ctx->h_positions;
        CUDA_CHECK(cudaFreeHost(ctx->h_n_pinned));
        CUDA_CHECK(cudaFreeHost(ctx->h_vxc_pinned));
        delete ctx;
    }
}

#endif // GPU_VXC_CU_H