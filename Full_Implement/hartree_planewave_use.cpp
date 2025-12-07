#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <fftw3.h>
#include <fstream>
#include <sstream>
#include <iomanip>

#define PI 3.14159265358979323846

namespace numint
{
    struct TimedResult
    {
        double value;  // e.g. Hartree potential at one grid point
        double time_s; // runtime in seconds
        fftw_complex* VH_g;
    };
    // creates Flattened linear index for grid
    inline int idx3D(int ix, int iy, int iz, int Ny, int Nz)
    {
        return (ix * Ny + iy) * Nz + iz;
    }
    // Maps indexes of FFt to kval?
    inline int fft_index_to_kint(int i, int N)
    {
        return (i <= N / 2) ? i : (i - N);
    }

    
    /*
    sequentail calculation of hartree potienail
    Ck_real, Ck_imag: plane-wave coefficients in reciprocal space (complex numbers split into real/imag parts), one per grid point in G-space.
    Lx, Ly, Lz: physical box size in x, y, z (so G = 2π n / L).
    Nx, Ny, Nz: grid resolution in each direction.
    ix_eval, iy_eval, iz_eval: which real-space grid point you want VH(r)

    */
    TimedResult Vh_PlaneWave_3D_s(const std::vector<double> &Ck_real, const std::vector<double> &Ck_imag, double Lx, double Ly, double Lz, int Nx, int Ny, int Nz, int ix_eval, int iy_eval, int iz_eval)
    {
        TimedResult results;

        using namespace std::chrono;
        auto t0 = high_resolution_clock::now(); // begin timing clock

        int Ntot = Nx * Ny * Nz;
        if ((int)Ck_real.size() != Ntot || (int)Ck_imag.size() != Ntot) // check that the grid is approprately sized and bounds are within grid
        {
            std::cerr << "Error: Ck_real and Ck_imag must be size Nx*Ny*Nz.\n";
            return {0.0, 0.0};
        }
        if (ix_eval < 0 || ix_eval >= Nx ||
            iy_eval < 0 || iy_eval >= Ny ||
            iz_eval < 0 || iz_eval >= Nz)
        {
            std::cerr << "Error: out of range.\n";
            return {0.0, 0.0};
        }

        const double two_pi = 2.0 * PI;

        // creates complex plains in K-space and real space for the wave functions, density functions and poteintals
        fftw_complex *psi_k = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Ntot);
        fftw_complex *psi_r = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Ntot);
        fftw_complex *n_r = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Ntot);
        fftw_complex *n_k = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Ntot);
        fftw_complex *VH_k = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Ntot);
        fftw_complex *VH_r = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Ntot);

        if (!psi_k || !psi_r || !n_r || !n_k || !VH_k || !VH_r)
        {
            std::cerr << "FFTW allocation failed.\n";
            return {0.0, 0.0};
        }

        // Fill psi_k from input coefficients
        for (int n = 0; n < Ntot; ++n)
        {
            psi_k[n][0] = Ck_real[n]; // real part
            psi_k[n][1] = Ck_imag[n]; // imag part
        }

        //   Create FFTW plans
        fftw_plan plan_k_to_r_psi = fftw_plan_dft_3d(Nx, Ny, Nz, psi_k, psi_r, FFTW_BACKWARD, FFTW_ESTIMATE);

        fftw_plan plan_r_to_k_n = fftw_plan_dft_3d(Nx, Ny, Nz, n_r, n_k, FFTW_FORWARD, FFTW_ESTIMATE);

        fftw_plan plan_k_to_r_VH = fftw_plan_dft_3d(Nx, Ny, Nz, VH_k, VH_r, FFTW_BACKWARD, FFTW_ESTIMATE);

        // psi(k) to psi(r) via inverse FFT
        fftw_execute(plan_k_to_r_psi);

        // Normalize inverse FFT (FFTW backward has factor Ntot)
        for (int n = 0; n < Ntot; ++n)
        {
            psi_r[n][0] /= Ntot;
            psi_r[n][1] /= Ntot;
        }

        //  calculate density function
        for (int n = 0; n < Ntot; ++n)
        {
            double re = psi_r[n][0];
            double im = psi_r[n][1];
            double dens = re * re + im * im;

            n_r[n][0] = dens; // real
            n_r[n][1] = 0.0;  // imag = 0
        }

        //   n(r) to n(G) FFT
        fftw_execute(plan_r_to_k_n);

        // Compute Hartree potential in G-space
        //          V_H(G) = 4pi / |G|^2 * n(G),   G != 0
        //          V_H(G=0) = 0
        for (int ix = 0; ix < Nx; ++ix)
        {
            int kx_int = fft_index_to_kint(ix, Nx);
            double Gx = two_pi * kx_int / Lx;

            for (int iy = 0; iy < Ny; ++iy)
            {
                int ky_int = fft_index_to_kint(iy, Ny);
                double Gy = two_pi * ky_int / Ly;

                for (int iz = 0; iz < Nz; ++iz)
                {
                    int kz_int = fft_index_to_kint(iz, Nz);
                    double Gz = two_pi * kz_int / Lz;

                    int n = idx3D(ix, iy, iz, Ny, Nz);

                    double G2 = Gx * Gx + Gy * Gy + Gz * Gz;

                    if (G2 == 0.0)
                    {
                        VH_k[n][0] = 0.0; // real
                        VH_k[n][1] = 0.0; // imag
                    }
                    else
                    {
                       
                        // find electron-electron poteintal
                      
                        double factor = 4.0 * PI / G2;
                        double re = n_k[n][0];
                        double im = n_k[n][1];

                        VH_k[n][0] = factor * re;
                        VH_k[n][1] = factor * im;

                        // find nuclear interaction poteintal and smooth result
                        double Z = 26.0;     // Iron nuclear charge
                        double sigma = 0.10; // smoothing width (bohr)
                        double smooth = exp(-0.25 * G2 * sigma * sigma);

                        // Add smoothed Coulomb nuclear term  -4πZ/G² e^{-(σG)²/4}
                        VH_k[n][0] += -4.0 * PI * Z * smooth / G2;
                    }
                }
            }
        }

        results.VH_g = VH_k;

        //  inverse FFT  V_H(G) to V_H(r)
        fftw_execute(plan_k_to_r_VH);

        // normalize inverse FFT
        for (int n = 0; n < Ntot; ++n)
        {
            VH_r[n][0] /= Ntot;
            VH_r[n][1] /= Ntot;
        }

        //  get value at requested grid point
        int n_eval = idx3D(ix_eval, iy_eval, iz_eval, Ny, Nz);
        double VH_at_r = VH_r[n_eval][0]; // real

        auto t1 = high_resolution_clock::now(); // stop clock and find total time
        std::chrono::duration<double> elapsed = t1 - t0;

        //  clear up mem
        fftw_destroy_plan(plan_k_to_r_psi);
        fftw_destroy_plan(plan_r_to_k_n);
        fftw_destroy_plan(plan_k_to_r_VH);

        fftw_free(psi_k);
        fftw_free(psi_r);
        fftw_free(n_r);
        fftw_free(n_k);
        fftw_free(VH_k);
        fftw_free(VH_r);

        results.value = (2 * VH_at_r);
        results.time_s = elapsed.count();

        return results;
    }
    /*
   parallel calculation of hartree potienail
   Ck_real, Ck_imag: plane-wave coefficients in reciprocal space (complex numbers split into real/imag parts), one per grid point in G-space.
   Lx, Ly, Lz: physical box size in x, y, z (so G = 2π n / L).
   Nx, Ny, Nz: grid resolution in each direction.
   ix_eval, iy_eval, iz_eval: which real-space grid point you want VH(r)

   function takes the wave fucntion expanded into plane waves,


   */
    TimedResult Vh_PlaneWave_3D_p(const std::vector<double> &Ck_real, const std::vector<double> &Ck_imag, double Lx, double Ly, double Lz, int Nx, int Ny, int Nz, int ix_eval, int iy_eval, int iz_eval)
    {
        TimedResult results;
        using namespace std::chrono;
        auto t0 = high_resolution_clock::now(); // begin timing clock

        int Ntot = Nx * Ny * Nz;
        if ((int)Ck_real.size() != Ntot || (int)Ck_imag.size() != Ntot) // check that the grid is approprately sized and bounds are within grid
        {
            std::cerr << "Error: Ck_real and Ck_imag must be size Nx*Ny*Nz.\n";
            return {0.0, 0.0};
        }
        if (ix_eval < 0 || ix_eval >= Nx ||
            iy_eval < 0 || iy_eval >= Ny ||
            iz_eval < 0 || iz_eval >= Nz)
        {
            std::cerr << "Error: out of range.\n";
            return {0.0, 0.0};
        }

        const double two_pi = 2.0 * PI;
        fftw_init_threads();
        fftw_plan_with_nthreads(8);

        // creates complex plains in K-space and real space for the wave functions, density functions and poteintals
        fftw_complex *psi_k = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Ntot);
        fftw_complex *psi_r = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Ntot);
        fftw_complex *n_r = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Ntot);
        fftw_complex *n_k = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Ntot);
        fftw_complex *VH_k = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Ntot);
        fftw_complex *VH_r = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Ntot);

        if (!psi_k || !psi_r || !n_r || !n_k || !VH_k || !VH_r)
        {
            std::cerr << "FFTW allocation failed.\n";
            return {0.0, 0.0};
        }

// make psi_k from input coefficients n parallell
#pragma omp parallel for
        for (int n = 0; n < Ntot; ++n)
        {
            psi_k[n][0] = Ck_real[n]; // real part
            psi_k[n][1] = Ck_imag[n]; // imag part
        }

        //   Create FFTW plans
        fftw_plan plan_k_to_r_psi = fftw_plan_dft_3d(Nx, Ny, Nz, psi_k, psi_r, FFTW_BACKWARD, FFTW_ESTIMATE);

        fftw_plan plan_r_to_k_n = fftw_plan_dft_3d(Nx, Ny, Nz, n_r, n_k, FFTW_FORWARD, FFTW_ESTIMATE);

        fftw_plan plan_k_to_r_VH = fftw_plan_dft_3d(Nx, Ny, Nz, VH_k, VH_r, FFTW_BACKWARD, FFTW_ESTIMATE);

        // psi(k) to psi(r) via inverse FFT
        fftw_execute(plan_k_to_r_psi);

// Normalize inverse FFT in parallel
#pragma omp parallel for
        for (int n = 0; n < Ntot; ++n)
        {
            psi_r[n][0] /= Ntot;
            psi_r[n][1] /= Ntot;
        }

//  calculate density function in paralell
#pragma omp parallel for
        for (int n = 0; n < Ntot; ++n)
        {
            double re = psi_r[n][0];
            double im = psi_r[n][1];
            double dens = re * re + im * im;

            n_r[n][0] = dens; // real
            n_r[n][1] = 0.0;  // imag = 0
        }

        //   n(r) to n(G) FFT
        fftw_execute(plan_r_to_k_n);

// Compute Hartree potential in G-space
//          V_H(G) = 4pi / |G|^2 * n(G) for   G != 0
//          V_H(G=0) = 0
#pragma omp parallel for collapse(3)
        for (int ix = 0; ix < Nx; ++ix)
        {
            for (int iy = 0; iy < Ny; ++iy)
            {
                for (int iz = 0; iz < Nz; ++iz)
                {

                    int kx_int = fft_index_to_kint(ix, Nx);
                    int ky_int = fft_index_to_kint(iy, Ny);
                    int kz_int = fft_index_to_kint(iz, Nz);

                    double Gx = two_pi * kx_int / Lx;
                    double Gy = two_pi * ky_int / Ly;
                    double Gz = two_pi * kz_int / Lz;

                    int n = idx3D(ix, iy, iz, Ny, Nz);
                    double G2 = Gx * Gx + Gy * Gy + Gz * Gz;

                    if (G2 == 0.0)
                    {
                        VH_k[n][0] = 0.0;
                        VH_k[n][1] = 0.0;
                    }
                    else
                    {
                       // find electron-electron poteintal
                      
                        double factor = 4.0 * PI / G2;
                        double re = n_k[n][0];
                        double im = n_k[n][1];

                        VH_k[n][0] = factor * re;
                        VH_k[n][1] = factor * im;

                        // find nuclear interaction poteintal and smooth result
                        double Z = 26.0;     // Iron nuclear charge
                        double sigma = 0.10; // smoothing width (bohr)
                        double smooth = exp(-0.25 * G2 * sigma * sigma);

                        // Add smoothed Coulomb nuclear term  -4πZ/G² e^{-(σG)²/4}
                        VH_k[n][0] += -4.0 * PI * Z * smooth / G2;
                    }
                }
            }
        }
        results.VH_g = VH_k;

        //  inverse FFT  V_H(G) to V_H(r)
        fftw_execute(plan_k_to_r_VH);

        // normalize inverse FFT in parallel
#pragma omp parallel for
        for (int n = 0; n < Ntot; ++n)
        {
            VH_r[n][0] /= Ntot;
            VH_r[n][1] /= Ntot;
        }

        //  get value at requested grid point
        int n_eval = idx3D(ix_eval, iy_eval, iz_eval, Ny, Nz);
        double VH_at_r = VH_r[n_eval][0]; // real part

        auto t1 = high_resolution_clock::now(); // stop clock and find total time
        std::chrono::duration<double> elapsed = t1 - t0;

        //  clear up mem
        fftw_destroy_plan(plan_k_to_r_psi);
        fftw_destroy_plan(plan_r_to_k_n);
        fftw_destroy_plan(plan_k_to_r_VH);

        fftw_free(psi_k);
        fftw_free(psi_r);
        fftw_free(n_r);
        fftw_free(n_k);
        fftw_free(VH_k);
        fftw_free(VH_r);

        results.value = (2 * VH_at_r);
        results.time_s = elapsed.count();

        return results;
    }

}

   
