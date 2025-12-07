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
#include <stdexcept>

#define PI 3.14159265358979323846

double compute_alpha_qe(
    double alat,                   // lattice parameter (bohr), QE "alat"
    const std::vector<double> &zv, // zv[0..ntyp-1] (ionic charges Zval)
    const std::vector<int> &ityp,  // ityp[0..nat-1] (type index for each atom, 0-based)
    double gcutm                   // max dimensionless |G|^2 (QE style)
)
{
    const double tpi = 2.0 * PI;
    const double tpiba2 = (tpi / alat) * (tpi / alat);

    // total ionic charge in the unit cell
    double charge = 0.0;
    for (int na = 0; na < (int)ityp.size(); ++na)
    {
        int t = ityp[na]; // type index
        charge += zv[t];
    }

    // start from alpha = 2.9 and decrease by 0.1 until upperbound < 1e-7
    double alpha = 2.9;
    while (true)
    {
        alpha -= 0.1;
        if (alpha <= 0.0)
        {
            throw std::runtime_error("ewald: optimal alpha not found (QE-style)");
        }

        double arg = std::sqrt(tpiba2 * gcutm / (4.0 * alpha));
        double upperbound = 2.0 * charge * charge * std::sqrt(2.0 * alpha / tpi) * std::erfc(arg);

        if (upperbound <= 1.0e-7)
            break;
    }

    return alpha; // dimensionless, QE-style
}

double compute_rmax_qe(double alpha, double alat)
{
    // rmax is in *crystal coordinate* units (multiplying at columns),
    // rr = sqrt(r2)*alat gives the real distance in bohr.
    return 4.0 / std::sqrt(alpha) / alat;
}

double compute_rmax_realspace_qe(double alpha)
{
    // rr such that sqrt(alpha)*rr = 4 -> rr = 4 / sqrt(alpha)
    return 4.0 / std::sqrt(alpha); // in bohr
}

static inline int index(int ix, int iy, int iz, int Nx, int Ny)
{
    return (iz * Ny + iy) * Nx + ix; // creates a 3d grid points
}

struct Vec3
{
    double x, y, z;
};

double Ve_PlaneWave_3D_s(
    const std::vector<Vec3> &ion_pos,
    const std::vector<double> &Z,
    double Lx, double Ly, double Lz,
    int Nx, int Ny, int Nz)
{
    double VE_at_r = 0;
    double alat = Lx;
    const int Ntot = Nx * Ny * Nz;
    const double V = Lx * Ly * Lz;

    double Gmax = 0.0;
    std::vector<double> zv = {Z};
    double alpha = 0;

    // inside your G loops:
    const double TWO_PI = 2.0 * PI;

    // compute ithe reciporacal ewald energy
    double E_rec = 0.0; // <<< CHANGED

    for (int iz = 0; iz < Nz; ++iz) // z-loop
    {
        int kz = (iz <= Nz / 2) ? iz : iz - Nz; // convert integer k values into G-vectors of form g=2pi/k and ensure that each value is signed properly such that
        double Gz = TWO_PI * kz / Lz;           // values range over full Bz so high frequncys are reflected back into the Bz as negitive frequencys such that they repesent the wave fucntion of prevouis perodic cell entering thsi cell

        for (int iy = 0; iy < Ny; ++iy)
        {
            int ky = (iy <= Ny / 2) ? iy : iy - Ny; // y-loop see above
            double Gy = TWO_PI * ky / Ly;

            for (int ix = 0; ix < Nx; ++ix)
            {
                int kx = (ix <= Nx / 2) ? ix : ix - Nx; // x-loop see above
                double Gx = TWO_PI * kx / Lx;

                int idx = index(ix, iy, iz, Nx, Ny); // creats a unit grid volume for each Nz

                double G = Gx * Gx + Gy * Gy + Gz * Gz; // calcualte G magnitude
                double Gmax = std::max(Gmax, 200.0 / (Lx * Lx));
                //Gmax = std::max(Gmax, G); // G is already |G|^2
                                          // double Gmax = std::max(Gmax, 200.0 / (Lx * Lx));

                // One species type, valence charge Zval
                double tpi_over_alat = 2.0 * PI / alat; // bohr^-1

                double gcutm = (Gmax / tpi_over_alat) * (Gmax / tpi_over_alat);

                std::vector<int> ityp(ion_pos.size(), 0); // all atoms are type 0
                alpha = compute_alpha_qe(alat, zv, ityp, gcutm);

                if (G == 0)
                { // ensure g=0 is 0
                    continue;
                }

                std::complex<double> Sg(0.0, 0.0); // create stucture factor vector

                // compute structure factor
                for (int i = 0; i < ion_pos.size(); ++i)
                {
                    double Gr_product = ion_pos[i].x * Gx + ion_pos[i].y * Gy + ion_pos[i].z * Gz; // compute dot product of G and R vectors
                    std::complex<double> phase(std::cos(Gr_product), -std::sin(Gr_product));       // convert dot product magnitude into a vector of real and imaginary magnitudes
                    Sg += Z[i] * phase;                                                            // sum the vectors such that we get sum(z_i*e^-iG*r)
                }

                double damping = std::exp(-G / (4.0 * alpha * alpha));

                E_rec += (4.0 * PI / G) * damping * std::norm(Sg); // calcutate gaussian smoothing factor for this G point
            }
        }
    }

    E_rec *= (1.0 / (2.0 * V)); // divide by 1/2v to renormailize and compneste for double suming giving the reciporacal space energy

    // calcualte realspace ewald energy
    double E_real = 0.0;
    double Rcut = compute_rmax_realspace_qe(alpha); // calculate cutoff radius for real space interatactions

    int NxR = int(Rcut / Lx) + 1; // for each dimention deterimine how many copies of the cell fit inside the cutoff radius
    int NyR = int(Rcut / Ly) + 1;
    int NzR = int(Rcut / Lz) + 1;

    for (int nx = -NxR; nx <= NxR; nx++) // loopnthrough each of the periodic cells by looping through each dim
        for (int ny = -NyR; ny <= NyR; ny++)
            for (int nz = -NzR; nz <= NzR; nz++)
            {

                double Rx = nx * Lx; // compute the translation vectors, ie the distance to shift the ions postion into the next cell
                double Ry = ny * Ly;
                double Rz = nz * Lz;

                for (int i = 0; i < ion_pos.size(); i++) // loop through each pair of ions
                    for (int j = 0; j < ion_pos.size(); j++)
                    {

                        if (nx == 0 && ny == 0 && nz == 0 && i == j) // ignore self ion interactions
                            continue;

                        double dx = ion_pos[i].x - ion_pos[j].x + Rx;
                        double dy = ion_pos[i].y - ion_pos[j].y + Ry;
                        double dz = ion_pos[i].z - ion_pos[j].z + Rz;
                        double r = std::sqrt(dx * dx + dy * dy + dz * dz);

                        if (r == 0.0)
                            continue;
                        if (r > Rcut)
                            continue; // ensure that the ion is in the radius

                        E_real += Z[i] * Z[j] * std::erfc(alpha * r) / r; // using the shortrange erfc aproximation to remove long range interactions
                    }
            }

    E_real *= 0.5; // correct for double sum

    double self_correction = 0;
    for (int i = 0; i < ion_pos.size(); i++)
    { // calcualte the self correting term
        self_correction += -alpha * Z[i] * Z[i] / std::sqrt(PI);
    }

    double backround = 0;
    for (double Zi : Z)
    { // sum the charge nuclear charge
        backround += Zi;
    }

    backround = -PI * (backround * backround) / (alpha * alpha * V); // background neutrality term

    return E_real + E_rec + self_correction + backround; // sum all componets
}

double Ve_PlaneWave_3D_P(
    const std::vector<Vec3> &ion_pos,
    const std::vector<double> &Z,
    double Lx, double Ly, double Lz,
    int Nx, int Ny, int Nz)
{
    double alat = Lx;
    const double V = Lx * Ly * Lz;
    const double TWO_PI = 2.0 * PI;

    // Pre-compute alpha BEFORE the parallel loop (thread-safe)
    double Gmax = 200.0 / (Lx * Lx);
    double tpi_over_alat = 2.0 * PI / alat;
    double gcutm = (Gmax / tpi_over_alat) * (Gmax / tpi_over_alat);
    std::vector<double> zv = Z;
    std::vector<int> ityp(ion_pos.size(), 0);
    double alpha = compute_alpha_qe(alat, zv, ityp, gcutm);

    // Compute the reciprocal ewald energy
    double E_rec = 0.0;

#pragma omp parallel for collapse(3) reduction(+ : E_rec)
    for (int iz = 0; iz < Nz; ++iz)
    {
        for (int iy = 0; iy < Ny; ++iy)
        {
            for (int ix = 0; ix < Nx; ++ix)
            {
                // Now compute all derived values inside the innermost loop
                int kz = (iz <= Nz / 2) ? iz : iz - Nz;
                double Gz = TWO_PI * kz / Lz;
                
                int ky = (iy <= Ny / 2) ? iy : iy - Ny;
                double Gy = TWO_PI * ky / Ly;
                
                int kx = (ix <= Nx / 2) ? ix : ix - Nx;
                double Gx = TWO_PI * kx / Lx;

                double G = Gx * Gx + Gy * Gy + Gz * Gz;

                if (G == 0)
                {
                    continue;
                }

                std::complex<double> Sg(0.0, 0.0);

                // compute structure factor
                for (int i = 0; i < ion_pos.size(); ++i)
                {
                    double Gr_product = ion_pos[i].x * Gx + ion_pos[i].y * Gy + ion_pos[i].z * Gz;
                    std::complex<double> phase(std::cos(Gr_product), -std::sin(Gr_product));
                    Sg += Z[i] * phase;
                }

                double damping = std::exp(-G / (4.0 * alpha * alpha));
                E_rec += (4.0 * PI / G) * damping * std::norm(Sg);
            }
        }
    }

    E_rec *= (1.0 / (2.0 * V));

    // Calculate realspace ewald energy
    double E_real = 0.0;
    double Rcut = compute_rmax_realspace_qe(alpha);

    int NxR = int(Rcut / Lx) + 1;
    int NyR = int(Rcut / Ly) + 1;
    int NzR = int(Rcut / Lz) + 1;

#pragma omp parallel for collapse(3) reduction(+ : E_real)
    for (int nx = -NxR; nx <= NxR; nx++)
    {
        for (int ny = -NyR; ny <= NyR; ny++)
        {
            for (int nz = -NzR; nz <= NzR; nz++)
            {
                double Rx = nx * Lx;
                double Ry = ny * Ly;
                double Rz = nz * Lz;

                for (int i = 0; i < ion_pos.size(); i++)
                {
                    for (int j = 0; j < ion_pos.size(); j++)
                    {
                        if (nx == 0 && ny == 0 && nz == 0 && i == j)
                            continue;

                        double dx = ion_pos[i].x - ion_pos[j].x + Rx;
                        double dy = ion_pos[i].y - ion_pos[j].y + Ry;
                        double dz = ion_pos[i].z - ion_pos[j].z + Rz;
                        double r = std::sqrt(dx * dx + dy * dy + dz * dz);

                        if (r == 0.0)
                            continue;
                        if (r > Rcut)
                            continue;

                        E_real += Z[i] * Z[j] * std::erfc(alpha * r) / r;
                    }
                }
            }
        }
    }

    E_real *= 0.5;

    double self_correction = 0;
    for (int i = 0; i < ion_pos.size(); i++)
    {
        self_correction += -alpha * Z[i] * Z[i] / std::sqrt(PI);
    }

    double backround = 0;
    for (double Zi : Z)
    {
        backround += Zi;
    }

    backround = -PI * (backround * backround) / (alpha * alpha * V);

    return E_real + E_rec + self_correction + backround;
}