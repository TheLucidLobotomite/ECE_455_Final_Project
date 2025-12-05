// ================================================================
// test_compare_hartree.cu
// Compare CPU FFTW Hartree solver vs CUDA/cuFFT solver
// ================================================================

#include <iostream>
#include <vector>
#include <cmath>
//#include <cuda_runtime.h>
#include <iomanip>
#include "hartree_planewave_use.hpp" // <-- Provides TimedResult

// Must declare: TimedResult Vh_PlaneWave_3D_s(...)
// GPU version (from hartree_planewave_cuda.cu)
numint::TimedResult Vh_PlaneWave_3D_cuda(const std::vector<double> &Ck_real,
                                 const std::vector<double> &Ck_imag,
                                 double Lx, double Ly, double Lz,
                                 int Nx, int Ny, int Nz,
                                 int ix_eval, int iy_eval, int iz_eval);

// ================================================================
// Main test
// ================================================================
int main()

{

    // Grid resolution
    int Nx = 16;
    int Ny = 16;
    int Nz = 16;
    int Ntot = Nx * Ny * Nz;

    // Cell dimensions
    double Lx = 10.0;
    double Ly = 10.0;
    double Lz = 10.0;

    // Evaluate Hartree potential at center grid point
    int ix_eval = Nx / 2;
    int iy_eval = Ny / 2;
    int iz_eval = Nz / 2;

    // Allocate plane-wave coefficients
    std::vector<double> Ck_real(Ntot);
    std::vector<double> Ck_imag(Ntot);

    // Fill Ck with a Gaussian in k-space
    double sigma = 2.0;

    for (int ix = 0; ix < Nx; ix++)
    {
        int kx = (ix <= Nx / 2) ? ix : ix - Nx;

        for (int iy = 0; iy < Ny; iy++)
        {
            int ky = (iy <= Ny / 2) ? iy : iy - Ny;

            for (int iz = 0; iz < Nz; iz++)
            {
                int kz = (iz <= Nz / 2) ? iz : iz - Nz;

                int n = (ix * Ny + iy) * Nz + iz;
                double k2 = kx * kx + ky * ky + kz * kz;
                double amp = std::exp(-k2 / (2.0 * sigma * sigma));

                Ck_real[n] = amp;
                Ck_imag[n] = 0.0;
            }
        }
    }

    std::cout << "\n============================================\n";
    std::cout << "     CPU FFTW vs CUDA Hartree Comparison\n";
    std::cout << "============================================\n";

    // ============================================================
    // Run CPU Serial version
    // ============================================================
    numint::TimedResult cpu_s = numint::Vh_PlaneWave_3D_s(
        Ck_real, Ck_imag, Lx, Ly, Lz,
        Nx, Ny, Nz,
        ix_eval, iy_eval, iz_eval);

    std::cout << "\nCPU Serial FFTW:\n";
    std::cout << "  Value = " << std::setprecision(12) << cpu_s.value << "\n";
    std::cout << "  Time  = " << cpu_s.time << " s\n";

   numint::TimedResult gpu = Vh_PlaneWave_3D_cuda(
        Ck_real, Ck_imag, Lx, Ly, Lz,
        Nx, Ny, Nz, ix_eval, iy_eval, iz_eval
    );

    std::cout << "\nCUDA/cuFFT:\n";
    std::cout << "  Value = " << std::setprecision(12) << gpu.value << "\n";
    std::cout << "  Time  = " << gpu.time << " s\n";


    // ------------------------------------------------------------
    // Compare numerical accuracy
    // ------------------------------------------------------------
    double denom = std::max(1e-12, std::abs(cpu_s.value));
    double rel_err = std::abs(cpu_s.value - gpu.value) / denom;

    std::cout << "\nRelative Error = " << rel_err << "\n";

    if (rel_err < 1e-6)
        std::cout << "MATCH ✔ (within tolerance)\n\n";
    else
        std::cout << "WARNING: mismatch exceeds tolerance \n\n";

    return 0;
}
