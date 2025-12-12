#include <iostream>
#include <vector>
#include <cmath>
#include <fftw3.h>
#include <omp.h>
#include <iostream>

// tests hartree implemntation code to see if it works corectly
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

// test function evaluate the hatree engergy  using the hartree poental fucntions above
double compute_hartree_energy(const double *rho_r,const double *vhart_r,int Nx, int Ny, int Nz,double Lx, double Ly, double Lz)
{
    const int Ntot = Nx * Ny * Nz;

    //compute realspace volume
    double dV = (Lx * Ly * Lz) / double(Ntot);

    // sum var for integral
    double sum = 0.0;
// numericly integrate rho*V_h using a reduction sum 
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < Ntot; i++)
    {
        sum += rho_r[i] * vhart_r[i];
    }

    // retrun hartree energy  1/2 integral rho* V_h* d^3 r
    return 0.5 * sum * dV;
}

// main test function, written by chat_gpt sligthly modifyed
int main()
{
    using namespace std;

    // -----------------------------------------------------------
    // BCC Iron parameters (QE typical)
    // -----------------------------------------------------------
    double a = 5.421; // bohr
    double Lx = a, Ly = a, Lz = a;

    // Atom positions in Bohr
    double ax = 0.0, ay = 0.0, az = 0.0;
    double bx = 0.5 * a, by = 0.5 * a, bz = 0.5 * a;

    // Grid resolution (QE often uses 24^3–48^3 for tests)
    int Nx = 40, Ny = 40, Nz = 40;
    int Ntot = Nx * Ny * Nz;

    // Allocate density
    vector<double> rho_r(Ntot, 0.0);

    // -----------------------------------------------------------
    // Build QE-like pseudocharge density with Gaussians
    // -----------------------------------------------------------
    double sigma = 0.45; // pseudopotential core width (adjustable)

    for (int ix = 0; ix < Nx; ix++)
    {
        double x = ix * (Lx / Nx);
        for (int iy = 0; iy < Ny; iy++)
        {
            double y = iy * (Ly / Ny);
            for (int iz = 0; iz < Nz; iz++)
            {
                double z = iz * (Lz / Nz);

                int i = numint::idx3D(ix, iy, iz, Ny, Nz);

                double r2a = (x - ax) * (x - ax) + (y - ay) * (y - ay) + (z - az) * (z - az);
                double r2b = (x - bx) * (x - bx) + (y - by) * (y - by) + (z - bz) * (z - bz);

                rho_r[i] = exp(-r2a / (2 * sigma * sigma)) + exp(-r2b / (2 * sigma * sigma));
            }
        }
    }

    // -----------------------------------------------------------
    // Allocate FFTW array + forward plan
    // -----------------------------------------------------------
    fftw_complex *rho_g = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Ntot);

    fftw_plan plan_fwd = fftw_plan_dft_3d(
        Nx, Ny, Nz,
        rho_g, rho_g,
        FFTW_FORWARD, FFTW_ESTIMATE);

    // Copy
    for (int i = 0; i < Ntot; i++)
    {
        rho_g[i][0] = rho_r[i];
        rho_g[i][1] = 0.0;
    }

    fftw_execute(plan_fwd);

    // Convert to vector form
    vector<double> Ck_real(Ntot), Ck_imag(Ntot);
    for (int i = 0; i < Ntot; i++)
    {
        Ck_real[i] = rho_g[i][0];
        Ck_imag[i] = rho_g[i][1];
    }

    // -----------------------------------------------------------
    // Compute V_H at both Fe atoms
    // -----------------------------------------------------------
    int ixA = Nx / 2;
    int iyA = Ny / 2;
    int izA = Nz / 2;

    int ixB = (int)round((0.5 * a) / (Lx / Nx));
    int iyB = (int)round((0.5 * a) / (Ly / Ny));
    int izB = (int)round((0.5 * a) / (Lz / Nz));

    auto resA = Vh_PlaneWave_3D_p(Ck_real, Ck_imag,
                                  Lx, Ly, Lz,
                                  Nx, Ny, Nz,
                                  ixA, iyA, izA);

    auto resB = Vh_PlaneWave_3D_p(Ck_real, Ck_imag,
                                  Lx, Ly, Lz,
                                  Nx, Ny, Nz,
                                  ixB, iyB, izB);

    cout << "\n===== QE-STYLE BCC IRON HARTREE TEST =====\n";
    cout << "Lattice constant a = " << a << " bohr\n";
    cout << "Grid = " << Nx << " x " << Ny << " x " << Nz << "\n";
    cout << "Sigma = " << sigma << " bohr\n\n";

    cout << "Hartree V_H at Fe (0,0,0):        " << resA.value << " Ry\n";
    cout << "Hartree V_H at Fe (a/2,a/2,a/2):  " << resB.value << " Ry\n";

    // Cleanup
    fftw_free(rho_g);
    fftw_free(resA.VH_g);
    fftw_free(resB.VH_g);
    fftw_destroy_plan(plan_fwd);

    return 0;
}