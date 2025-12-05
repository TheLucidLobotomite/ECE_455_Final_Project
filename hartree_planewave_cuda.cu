// ======================================================
// hartree_planewave_cuda.cu
// CUDA implementation of plane-wave Hartree potential
// ======================================================

#include <iostream>
#include <vector>
#include <cmath>
#include "hartree_planewave_use.hpp"
#include <cuda_runtime.h>
#include <cufft.h>

// ======================================================
// Result struct (same interface as CPU version)
// ======================================================
struct TimedResult
{
    double value;
    double time; // seconds
};

// ======================================================
// Device helper functions
// ======================================================

__device__ __forceinline__ int fft_index_to_kint_dev(int i, int N)
{
    return (i <= N / 2) ? i : i - N;
}

__device__ __forceinline__ int idx3D_dev(int ix, int iy, int iz, int Ny, int Nz)
{
    return (ix * Ny + iy) * Nz + iz;
}

// ======================================================
// CUDA kernels
// ======================================================

// Fill psi_k from real/imag arrays
__global__ void fillPsiKKernel(cufftDoubleComplex *psi_k,
                               const double *Ck_real,
                               const double *Ck_imag,
                               int Ntot)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= Ntot)
        return;
    psi_k[n].x = Ck_real[n];
    psi_k[n].y = Ck_imag[n];
}

// Normalize complex array by dividing by norm
__global__ void normalizeKernel(cufftDoubleComplex *data, double norm, int Ntot)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= Ntot)
        return;
    data[n].x /= norm;
    data[n].y /= norm;
}

// Compute density n_r = |psi_r|^2
__global__ void densityKernel(const cufftDoubleComplex *psi_r,
                              cufftDoubleComplex *n_r,
                              int Ntot)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= Ntot)
        return;

    double re = psi_r[n].x;
    double im = psi_r[n].y;

    double dens = re * re + im * im;

    n_r[n].x = dens;
    n_r[n].y = 0.0;
}

// Compute V_H(G) kernel
__global__ void hartreeKernel(const cufftDoubleComplex *n_k,
                              cufftDoubleComplex *VH_k,
                              int Nx, int Ny, int Nz,
                              double Lx, double Ly, double Lz,
                              double Z, double sigma)
{
    int Ntot = Nx * Ny * Nz;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= Ntot)
        return;

    int NyNz = Ny * Nz;

    int ix = n / NyNz;
    int rem = n - ix * NyNz;
    int iy = rem / Nz;
    int iz = rem - iy * Nz;

    const double two_pi = 2.0 * M_PI;

    int kx_int = fft_index_to_kint_dev(ix, Nx);
    int ky_int = fft_index_to_kint_dev(iy, Ny);
    int kz_int = fft_index_to_kint_dev(iz, Nz);

    double Gx = two_pi * kx_int / Lx;
    double Gy = two_pi * ky_int / Ly;
    double Gz = two_pi * kz_int / Lz;

    double G2 = Gx * Gx + Gy * Gy + Gz * Gz;

    cufftDoubleComplex out;

    if (G2 == 0.0)
    {
        out.x = 0.0;
        out.y = 0.0;
    }
    else
    {
        cufftDoubleComplex nk = n_k[n];
        double factor = 4.0 * M_PI / G2;

        // Electron–electron Hartree
        out.x = factor * nk.x;
        out.y = factor * nk.y;

        // Smooth nuclear term
        double smooth = exp(-0.25 * G2 * sigma * sigma);

        out.x += -4.0 * M_PI * Z * smooth / G2;
        // imag part stays
    }

    VH_k[n] = out;
}

// ======================================================
// Main CUDA Hartree function
// ======================================================

TimedResult Vh_PlaneWave_3D_cuda(
    const std::vector<double> &Ck_real,
    const std::vector<double> &Ck_imag,
    double Lx, double Ly, double Lz,
    int Nx, int Ny, int Nz,
    int ix_eval, int iy_eval, int iz_eval)
{
    TimedResult result = {0.0, 0.0};

    int Ntot = Nx * Ny * Nz;

    // ----------------------------------
    // CUDA timing
    // ----------------------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // ----------------------------------
    // Allocate device memory
    // ----------------------------------
    cufftDoubleComplex
        *d_psi_k,
        *d_psi_r,
        *d_n_r, *d_n_k,
        *d_VH_k, *d_VH_r;

    double *d_Ck_real, *d_Ck_imag;

    size_t cbytes = Ntot * sizeof(cufftDoubleComplex);
    size_t rbytes = Ntot * sizeof(double);

    cudaMalloc(&d_psi_k, cbytes);
    cudaMalloc(&d_psi_r, cbytes);
    cudaMalloc(&d_n_r, cbytes);
    cudaMalloc(&d_n_k, cbytes);
    cudaMalloc(&d_VH_k, cbytes);
    cudaMalloc(&d_VH_r, cbytes);

    cudaMalloc(&d_Ck_real, rbytes);
    cudaMalloc(&d_Ck_imag, rbytes);

    // Copy C(k) to device
    cudaMemcpy(d_Ck_real, Ck_real.data(), rbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ck_imag, Ck_imag.data(), rbytes, cudaMemcpyHostToDevice);

    // ----------------------------------
    // Fill psi_k
    // ----------------------------------
    int block = 256;
    int grid = (Ntot + block - 1) / block;

    fillPsiKKernel<<<grid, block>>>(d_psi_k, d_Ck_real, d_Ck_imag, Ntot);
    cudaDeviceSynchronize();

    // ----------------------------------
    // cuFFT plans
    // ----------------------------------
    cufftHandle planPsi, planN, planVH;

    cufftPlan3d(&planPsi, Nx, Ny, Nz, CUFFT_Z2Z);
    cufftPlan3d(&planN, Nx, Ny, Nz, CUFFT_Z2Z);
    cufftPlan3d(&planVH, Nx, Ny, Nz, CUFFT_Z2Z);

    // ----------------------------------
    // 1) psi(k) → psi(r) (inverse FFT)
    // ----------------------------------
    cufftExecZ2Z(planPsi, d_psi_k, d_psi_r, CUFFT_INVERSE);

    normalizeKernel<<<grid, block>>>(d_psi_r, double(Ntot), Ntot);

    // ----------------------------------
    // 2) density n(r)
    // ----------------------------------
    densityKernel<<<grid, block>>>(d_psi_r, d_n_r, Ntot);

    // ----------------------------------
    // 3) n(r) → n(G)
    // ----------------------------------
    cufftExecZ2Z(planN, d_n_r, d_n_k, CUFFT_FORWARD);

    // ----------------------------------
    // 4) Compute V_H(G)
    // ----------------------------------
    double Z = 26.0;
    double sigma = 0.10;

    hartreeKernel<<<grid, block>>>(d_n_k, d_VH_k,
                                   Nx, Ny, Nz,
                                   Lx, Ly, Lz,
                                   Z, sigma);

    cudaDeviceSynchronize();

    // ----------------------------------
    // 5) V_H(G) → V_H(r)
    // ----------------------------------
    cufftExecZ2Z(planVH, d_VH_k, d_VH_r, CUFFT_INVERSE);

    normalizeKernel<<<grid, block>>>(d_VH_r, double(Ntot), Ntot);

    // ----------------------------------
    // Read final result at requested r index
    // ----------------------------------
    int n_eval = (ix_eval * Ny + iy_eval) * Nz + iz_eval;
    cufftDoubleComplex host_val;

    cudaMemcpy(&host_val, d_VH_r + n_eval,
               sizeof(cufftDoubleComplex),
               cudaMemcpyDeviceToHost);

    result.value = 2.0 * host_val.x; // match CPU convention

    // ----------------------------------
    // Stop timer
    // ----------------------------------
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    result.time = ms / 1000.0;

    // ----------------------------------
    // Cleanup
    // ----------------------------------
    cufftDestroy(planPsi);
    cufftDestroy(planN);
    cufftDestroy(planVH);

    cudaFree(d_psi_k);
    cudaFree(d_psi_r);
    cudaFree(d_n_r);
    cudaFree(d_n_k);
    cudaFree(d_VH_k);
    cudaFree(d_VH_r);

    cudaFree(d_Ck_real);
    cudaFree(d_Ck_imag);

    return result;
}
