#include <iostream>
#include <vector>
#include <cmath>
#include <fftw3.h>
#include <omp.h>
#include <iostream>
namespace numint
{

    // creates flattened linear index for grid
    inline int idx3D(int ix, int iy, int iz, int Ny, int Nz)
    {
        return (ix * Ny + iy) * Nz + iz;
    }
    // Maps indexes of FFt to each kval
    inline int fft_index_to_kint(int i, int N)
    {
        return (i <= N / 2) ? i : (i - N);
    }

    struct V_h
    {
        double value;       // Hartree potential at evaluation point
        fftw_complex *VH_g; // Fourier coefficients V_H(G)
    };

}
/*
   sequentail calculation of hartree potienail
   rho_real, rho_imag: plane-wave coefficients in reciprocal space, real and imaginary compents of the density
   Lx, Ly, Lz: physical box size in x, y, z (so G = 2pi n / L)
   Nx, Ny, Nz: grid resolution in each direction
   ix_pt, iy_pt, iz_pt: real-space grid point for r val

   calcualte the hartree poteinal using the the posion eq V_h= 4pi/G^2*rho(r) and return a complex vector of potentals for the grid and the poteinal at a spesifc pt
   */
numint::V_h Vh_PlaneWave_3D_s(const std::vector<double> &rho_real, const std::vector<double> &rho_imag, double Lx, double Ly, double Lz, int Nx, int Ny, int Nz, int ix_pt, int iy_pt, int iz_pt)
{
    using namespace std;

    // calculate grid/cell size
    const int Ntot = Nx * Ny * Nz;

    // create hartree potienal comlpex plane
    fftw_complex *VH_g = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Ntot);

    // convert G(0) points to kvals of form k=2pi/l
    const double gx0 = 2.0 * M_PI / Lx;
    const double gy0 = 2.0 * M_PI / Ly;
    const double gz0 = 2.0 * M_PI / Lz;

    // Compute hartree poteinalin using the posion eq V_h= 4pi/G^2*rho(r)

    for (int ix = 0; ix < Nx; ix++)
    { // for x axis map each x-point to kspace point
        int kx = numint::fft_index_to_kint(ix, Nx);
        double gx = kx * gx0; // convert kval into reicptacal latice vector

        for (int iy = 0; iy < Ny; iy++)
        { // for y axis map each y-point to kspace point for this x pt
            int ky = numint::fft_index_to_kint(iy, Ny);
            double gy = ky * gy0; // convert kval into reicptacal latice vector

            for (int iz = 0; iz < Nz; iz++)
            { // for x axis map each z-point to kspace point for this y pt and xpt
                int kz = numint::fft_index_to_kint(iz, Nz);
                double gz = kz * gz0; // convert kval into reicptacal latice vector

                int i = numint::idx3D(ix, iy, iz, Ny, Nz); // map the grid point to an indaviual integer value

                double G = gx * gx + gy * gy + gz * gz; // find the squared magintiude of the  recipracal vector

                if (G > 1e-14)
                { // for non zero G vals calcualte the hartree potienatal for each pt per V_h= 4pi/G^2*rho(r)
                    double scale = 4.0 * M_PI / G;
                    VH_g[i][0] = scale * rho_real[i];
                    VH_g[i][1] = scale * rho_imag[i];
                }
                else
                {
                    // set g=0 for bcc
                    VH_g[i][0] = 0.0;
                    VH_g[i][1] = 0.0;
                }
            }
        }
    }

    // calculate hartree poteinal at evaluation point
    double VH_pt = 0.0;

    for (int ix = 0; ix < Nx; ix++)
    { // for x axis map each x-point to kspace point
        int kx = numint::fft_index_to_kint(ix, Nx);
        double gx = kx * gx0; // convert kval into reicptacal latice vector

        for (int iy = 0; iy < Ny; iy++)
        { // for y axis map each y-point to kspace point for this x pt
            int ky = numint::fft_index_to_kint(iy, Ny);
            double gy = ky * gy0; // convert kval into reicptacal latice vector

            for (int iz = 0; iz < Nz; iz++)
            { // for x axis map each z-point to kspace point for this y pt and xpt
                int kz = numint::fft_index_to_kint(iz, Nz);
                double gz = kz * gz0; // convert kval into reicptacal latice vector

                int i = numint::idx3D(ix, iy, iz, Ny, Nz); // map the grid point to an indaviual integer value

                double x = ix_pt * (Lx / Nx); // convert each grid index to real space val for the evaluation pt
                double y = iy_pt * (Ly / Ny);
                double z = iz_pt * (Lz / Nz);

                double phase = gx * x + gy * y + gz * z; // find the complex phase of the G val

                double real = cos(phase); // find the real componet of the phase
                double img = sin(phase);  // find the imaginary veraion of the phase

                VH_pt += VH_g[i][0] * real - VH_g[i][1] * img; // add the real and imaginary compontets of the the potental to the V_H array
            }
        }
    }

    // normalize point to size of FFT
    VH_pt /= (double)Ntot;

    // return value at evaluation pt and array of V_H in recipracal space
    return {VH_pt, VH_g};
}

/*
parallel calculation of hartree potienail
rho_real, rho_imag: plane-wave coefficients in reciprocal space, real and imaginary compents of the density
Lx, Ly, Lz: physical box size in x, y, z (so G = 2pi n / L)
Nx, Ny, Nz: grid resolution in each direction
ix_pt, iy_pt, iz_pt: real-space grid point for r val

calcualte the hartree poteinal using the the posion eq V_h= 4pi/G^2*rho(r) and return a complex vector of potentals for the grid and the poteinal at a spesifc pt
*/

numint::V_h Vh_PlaneWave_3D_p(const std::vector<double> &rho_real, const std::vector<double> &rho_imag, double Lx, double Ly, double Lz, int Nx, int Ny, int Nz, int ix_pt, int iy_pt, int iz_pt)
{
    using namespace std;

    // calculate grid/cell size
    const int Ntot = Nx * Ny * Nz;

    // create hartree potienal comlpex plane
    fftw_complex *VH_g = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Ntot);

    // convert G(0) points to kvals of form k=2pi/l
    const double gx0 = 2.0 * M_PI / Lx;
    const double gy0 = 2.0 * M_PI / Ly;
    const double gz0 = 2.0 * M_PI / Lz;

// ompute hartree poteinalin using the posion eq V_h= 4pi/G^2*rho(r) in parallel
#pragma omp parallel for
    for (int ix = 0; ix < Nx; ix++)
    { // for x axis map each x-point to kspace point
        int kx = numint::fft_index_to_kint(ix, Nx);
        double gx = kx * gx0; // convert kval into reicptacal latice vector

        for (int iy = 0; iy < Ny; iy++)
        { // for y axis map each y-point to kspace point for this x pt
            int ky = numint::fft_index_to_kint(iy, Ny);
            double gy = ky * gy0; // convert kval into reicptacal latice vector

            for (int iz = 0; iz < Nz; iz++)
            { // for x axis map each z-point to kspace point for this y pt and xpt
                int kz = numint::fft_index_to_kint(iz, Nz);
                double gz = kz * gz0; // convert kval into reicptacal latice vector

                int i = numint::idx3D(ix, iy, iz, Ny, Nz); // map the grid point to an indaviual integer value

                double G = gx * gx + gy * gy + gz * gz; // find the squared magintiude of the  recipracal vector

                if (G > 1e-14)
                { // for non zero G vals calcualte the hartree potienatal for each pt per V_h= 4pi/G^2*rho(r)
                    double scale = 4.0 * M_PI / G;
                    VH_g[i][0] = scale * rho_real[i];
                    VH_g[i][1] = scale * rho_imag[i];
                }
                else
                {
                    // set g=0 for bcc
                    VH_g[i][0] = 0.0;
                    VH_g[i][1] = 0.0;
                }
            }
        }
    }

    // calculate hartree poteinal at evalualtin point in parallel
    double VH_pt = 0.0;

#pragma omp parallel for reduction(+ : VH_pt)
    for (int ix = 0; ix < Nx; ix++)
    { // for x axis map each x-point to kspace point
        int kx = numint::fft_index_to_kint(ix, Nx);
        double gx = kx * gx0; // convert kval into reicptacal latice vector

        for (int iy = 0; iy < Ny; iy++)
        { // for y axis map each y-point to kspace point for this x pt
            int ky = numint::fft_index_to_kint(iy, Ny);
            double gy = ky * gy0; // convert kval into reicptacal latice vector

            for (int iz = 0; iz < Nz; iz++)
            { // for x axis map each z-point to kspace point for this y pt and xpt
                int kz = numint::fft_index_to_kint(iz, Nz);
                double gz = kz * gz0; // convert kval into reicptacal latice vector

                int i = numint::idx3D(ix, iy, iz, Ny, Nz); // map the grid point to an indaviual integer value

                double x = ix_pt * (Lx / Nx); // convert each grid index to real space val for the evaluation pt
                double y = iy_pt * (Ly / Ny);
                double z = iz_pt * (Lz / Nz);

                double phase = gx * x + gy * y + gz * z; // find the complex phase of the G val

                double real = cos(phase); // find the real componet of the phase
                double img = sin(phase);  // find the imaginary veraion of the phase

                VH_pt += VH_g[i][0] * real - VH_g[i][1] * img; // add the real and imaginary compontets of the the potental to the V_H array
            }
        }
    }
    // normalize point to size of FFT
    VH_pt /= (double)Ntot;

    // return value at evaluation pt and array of V_H in recipracal space
    return {VH_pt, VH_g};
}