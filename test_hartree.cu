#include <iostream>
#include <vector>
#include <cmath>

#include <cuda_runtime.h>
#include <cufft.h>

// ------------------------------------------------------------
// Insert or #include your code containing:
// TimedResult { double value; double time; };
// Vh_PlaneWave_3D_cuda(...)
// ------------------------------------------------------------

struct TimedResult {
    double value;
    double time;
};

// Declare your CUDA function
TimedResult Vh_PlaneWave_3D_cuda(const std::vector<double> &Ck_real,
                                 const std::vector<double> &Ck_imag,
                                 double Lx, double Ly, double Lz,
                                 int Nx, int Ny, int Nz,
                                 int ix_eval, int iy_eval, int iz_eval);


// ------------------------------------------------------------
// Main test driver
// ------------------------------------------------------------
int main()
{
    // Small test grid
    int Nx = 16;
    int Ny = 16;
    int Nz = 16;
    int Ntot = Nx * Ny * Nz;

    double Lx = 10.0;
    double Ly = 10.0;
    double Lz = 10.0;

    // Evaluate Hartree potential at the center
    int ix_eval = Nx / 2;
    int iy_eval = Ny / 2;
    int iz_eval = Nz / 2;

    // Allocate C(k) coefficients
    std::vector<double> Ck_real(Ntot);
    std::vector<double> Ck_imag(Ntot);

    // Simple test: C(k) = Gaussian-like function in k-space
    double sigma = 2.0;

    for (int ix = 0; ix < Nx; ix++)
    {
        int kx = (ix <= Nx/2) ? ix : ix - Nx;

        for (int iy = 0; iy < Ny; iy++)
        {
            int ky = (iy <= Ny/2) ? iy : iy - Ny;

            for (int iz = 0; iz < Nz; iz++)
            {
                int kz = (iz <= Nz/2) ? iz : iz - Nz;

                double k2 = kx*kx + ky*ky + kz*kz;
                double amp = exp(-k2 / (2.0 * sigma * sigma));

                int n = (ix * Ny + iy) * Nz + iz;

                Ck_real[n] = amp;
                Ck_imag[n] = 0.0;
            }
        }
    }

    std::cout << "Running CUDA Hartree test on "
              << Nx << "×" << Ny << "×" << Nz << " grid...\n";

    // Call your CUDA function
    TimedResult result = Vh_PlaneWave_3D_cuda(
        Ck_real, Ck_imag,
        Lx, Ly, Lz,
        Nx, Ny, Nz,
        ix_eval, iy_eval, iz_eval
    );

    std::cout << "\n=== CUDA Hartree Test Result ===\n";
    std::cout << "Hartree potential at center = " << result.value << "\n";
    std::cout << "GPU compute time = " << result.time << " seconds\n";

    return 0;
}
