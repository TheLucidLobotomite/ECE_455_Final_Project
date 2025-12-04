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

namespace numint
{
    struct TimedResult
    {
        double value;  // e.g. Hartree potential at one grid point
        double time_s; // runtime in seconds
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

    // ---------------------------------------------------------------
    // 1) Read DOS from CSV (columns: E, N)
    // ---------------------------------------------------------------
    void read_dos_csv(const std::string &filename,
                      std::vector<double> &E,
                      std::vector<double> &DOS)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open " << filename << "\n";
            std::exit(1);
        }

        std::string line;

        // skip header
        std::getline(file, line);

        while (std::getline(file, line))
        {
            if (line.empty())
                continue;

            std::stringstream ss(line);
            std::string valE, valN;

            std::getline(ss, valE, ',');
            std::getline(ss, valN, ',');

            if (!valE.empty() && !valN.empty())
            {
                E.push_back(std::stod(valE));
                DOS.push_back(std::stod(valN));
            }
        }
    }

    // ---------------------------------------------------------------
    // 2) Build cumulative DOS N(E) = ∫ DOS dE
    // ---------------------------------------------------------------
    void build_cumulative_dos(const std::vector<double> &E,
                              const std::vector<double> &DOS,
                              std::vector<double> &Ncum)
    {
        int n = E.size();
        Ncum.resize(n);
        Ncum[0] = 0.0;

        for (int i = 1; i < n; ++i)
        {
            double dE = E[i] - E[i - 1];
            double avg = 0.5 * (DOS[i] + DOS[i - 1]);
            Ncum[i] = Ncum[i - 1] + avg * dE;
        }
    }

    // ---------------------------------------------------------------
    // 3) Build a normalized 3D Gaussian ψ(r) with total charge Ne
    // ---------------------------------------------------------------
    void build_gaussian_psi_r_3D(int Nx, int Ny, int Nz,
                                 double Lx, double Ly, double Lz,
                                 double Ne,
                                 std::vector<std::complex<double>> &psi_r)
    {
        int Ntot = Nx * Ny * Nz;
        psi_r.assign(Ntot, std::complex<double>(0.0, 0.0));

        double dx = Lx / Nx;
        double dy = Ly / Ny;
        double dz = Lz / Nz;

        double x0 = 0.5 * Lx;
        double y0 = 0.5 * Ly;
        double z0 = 0.5 * Lz;

        double sigma = Lx / 10.0;
        double sigma2 = sigma * sigma;

        double norm_sum = 0.0;

        for (int ix = 0; ix < Nx; ++ix)
        {
            double x = (ix + 0.5) * dx;
            for (int iy = 0; iy < Ny; ++iy)
            {
                double y = (iy + 0.5) * dy;
                for (int iz = 0; iz < Nz; ++iz)
                {
                    double z = (iz + 0.5) * dz;

                    double dx0 = x - x0;
                    double dy0 = y - y0;
                    double dz0 = z - z0;
                    double r2 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0;

                    double val = std::exp(-r2 / (2.0 * sigma2));

                    int n = idx3D(ix, iy, iz, Ny, Nz);
                    psi_r[n] = std::complex<double>(val, 0.0);

                    norm_sum += val * val;
                }
            }
        }

        // Normalize so ∫ |ψ|² dV = Ne
        double dV = dx * dy * dz;
        norm_sum *= dV;

        double scale = std::sqrt(Ne / norm_sum);

        for (int n = 0; n < Ntot; ++n)
            psi_r[n] *= scale;
    }

    // ---------------------------------------------------------------
    // 4) FFT ψ(r) → ψ(k) into Ck_real / Ck_imag
    // ---------------------------------------------------------------
    void psi_r_to_Ck(const std::vector<std::complex<double>> &psi_r,
                     int Nx, int Ny, int Nz,
                     std::vector<double> &Ck_real,
                     std::vector<double> &Ck_imag)
    {
        int Ntot = Nx * Ny * Nz;

        fftw_complex *psi_r_fftw = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Ntot);
        fftw_complex *psi_k_fftw = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Ntot);

        if (!psi_r_fftw || !psi_k_fftw)
        {
            std::cerr << "FFTW allocation failed.\n";
            std::exit(1);
        }

        // copy data
        for (int n = 0; n < Ntot; ++n)
        {
            psi_r_fftw[n][0] = psi_r[n].real();
            psi_r_fftw[n][1] = psi_r[n].imag();
        }

        // forward FFT
        fftw_plan plan = fftw_plan_dft_3d(Nx, Ny, Nz,
                                          psi_r_fftw, psi_k_fftw,
                                          FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);

        Ck_real.resize(Ntot);
        Ck_imag.resize(Ntot);

        for (int n = 0; n < Ntot; ++n)
        {
            Ck_real[n] = psi_k_fftw[n][0];
            Ck_imag[n] = psi_k_fftw[n][1];
        }

        fftw_free(psi_r_fftw);
        fftw_free(psi_k_fftw);
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

        const double two_pi = 2.0 * M_PI;

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
                      
                        double factor = 4.0 * M_PI / G2;
                        double re = n_k[n][0];
                        double im = n_k[n][1];

                        VH_k[n][0] = factor * re;
                        VH_k[n][1] = factor * im;

                        // find nuclear interaction poteintal and smooth result
                        double Z = 26.0;     // Iron nuclear charge
                        double sigma = 0.10; // smoothing width (bohr)
                        double smooth = exp(-0.25 * G2 * sigma * sigma);

                        // Add smoothed Coulomb nuclear term  -4πZ/G² e^{-(σG)²/4}
                        VH_k[n][0] += -4.0 * M_PI * Z * smooth / G2;
                    }
                }
            }
        }

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

        return {2 * VH_at_r, elapsed.count()};
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

        const double two_pi = 2.0 * M_PI;
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
                      
                        double factor = 4.0 * M_PI / G2;
                        double re = n_k[n][0];
                        double im = n_k[n][1];

                        VH_k[n][0] = factor * re;
                        VH_k[n][1] = factor * im;

                        // find nuclear interaction poteintal and smooth result
                        double Z = 26.0;     // Iron nuclear charge
                        double sigma = 0.10; // smoothing width (bohr)
                        double smooth = exp(-0.25 * G2 * sigma * sigma);

                        // Add smoothed Coulomb nuclear term  -4πZ/G² e^{-(σG)²/4}
                        VH_k[n][0] += -4.0 * M_PI * Z * smooth / G2;
                    }
                }
            }
        }
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

        return {2 * VH_at_r, elapsed.count()};
    }

}

   
void test_fe_atom_qe_matching(int threads)
{
    using namespace std;
    using namespace numint;

    omp_set_num_threads(threads);
    fftw_init_threads();
    fftw_plan_with_nthreads(threads);

   //generate grid of size 20 bohr radius and with 128 steps in each direction
    double L = 20.0;     // bohr
    double Lx = L, Ly = L, Lz = L;

    int Nx = 160, Ny = 160, Nz = 160;
    int Ntot = Nx * Ny * Nz; // total grid volume

    double dx = Lx / Nx;
    double dy = Ly / Ny;
    double dz = Lz / Nz;


    //define iron pseudo-valence densitys 
    double A1 = 3.70, a1 = 1.50;
    double A2 = 2.20, a2 = 0.45;
    double A3 = 1.10, a3 = 0.13;

    vector<double> psi_r(Ntot);
    // create vector for wavefunctions
    #pragma omp parallel for collapse(3) // populate psi indexes with planewave wave fucntions from densitys 
    for (int ix = 0; ix < Nx; ix++)
    for (int iy = 0; iy < Ny; iy++)
    for (int iz = 0; iz < Nz; iz++)
    {
        double x = (ix - Nx/2.0) * dx;
        double y = (iy - Ny/2.0) * dy;
        double z = (iz - Nz/2.0) * dz;

        double r = sqrt(x*x + y*y + z*z);

        double rho =
            A1 * exp(-a1*r) +
            A2 * exp(-a2*r) +
            A3 * exp(-a3*r);

        psi_r[idx3D(ix,iy,iz,Ny,Nz)] = sqrt(rho);
    }


    // generate complex grids for wave fucntiosn
    fftw_complex *psi_k = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*Ntot);
    fftw_complex *psi_r_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*Ntot);

    //populate psi grid in regualr space for real and imaginary coef.s
    for (int n = 0; n < Ntot; n++) {
        psi_r_c[n][0] = psi_r[n];
        psi_r_c[n][1] = 0.0;
    }

    //transform the wavefucntions into k-space and clean up planes
    fftw_plan p_r2k = fftw_plan_dft_3d(
        Nx,Ny,Nz,
        psi_r_c, psi_k,
        FFTW_FORWARD,
        FFTW_ESTIMATE);

    fftw_execute(p_r2k);
    fftw_destroy_plan(p_r2k);

    // populate the plane wave coeff. vectors in k-space
    vector<double> Ck_real(Ntot), Ck_imag(Ntot);
    for (int n = 0; n < Ntot; n++) {
        Ck_real[n] = psi_k[n][0];
        Ck_imag[n] = psi_k[n][1];
    }

    fftw_free(psi_k);
    fftw_free(psi_r_c);

// using the hartee poteintal functions find the value of the poteintal in Ry and average speed up over several runs
    int ix0 = Nx/2, iy0 = Ny/2, iz0 = Nz/2;
    const int Niter = 50;

    double VH_s_sum = 0.0;
    double VH_p_sum = 0.0;
    double Ts_sum   = 0.0;
    double Tp_sum   = 0.0;

    for (int i = 0; i < Niter; i++)
    {
        TimedResult S =
            Vh_PlaneWave_3D_s(Ck_real, Ck_imag,
                              Lx, Ly, Lz,
                              Nx, Ny, Nz,
                              ix0, iy0, iz0);

        TimedResult P =
            Vh_PlaneWave_3D_p(Ck_real, Ck_imag,
                              Lx, Ly, Lz,
                              Nx, Ny, Nz,
                              ix0, iy0, iz0);

        VH_s_sum += S.value;
        VH_p_sum += P.value;
        Ts_sum   += S.time_s;
        Tp_sum   += P.time_s;
    }
// find average speed up
    double VH_s_avg = VH_s_sum / Niter;
    double VH_p_avg = VH_p_sum / Niter;
    double Ts_avg   = Ts_sum / Niter;
    double Tp_avg   = Tp_sum / Niter;

//print results
    cout << "\n=========== Fe Atomic Test (QE-Matching, Averaged) ===========\n";
    cout << "threads = " << threads << "\n\n";
    cout << "Grid: " << Nx << " x " << Ny << " x " << Nz << "\n";
    cout << "Averaged over N = " << Niter << " runs.\n\n";

    cout << "V_H(center) sequential = " << VH_s_avg << " Ry\n";
    cout << "V_H(center) parallel   = " << VH_p_avg << " Ry\n\n";

    cout << "Avg time sequential = " << Ts_avg << " s\n";
    cout << "Avg time parallel   = " << Tp_avg << " s\n";
    cout << "Speedup = " << Ts_avg / Tp_avg << "x\n";
    cout << "==============================================================\n";
}



int main()
{
    // run iron test
    using namespace numint;
    for(int i=2; i<17;i+=2){
        
            test_fe_atom_qe_matching(i);
        
    }
    

 
}
