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

double compute_alpha_qe(
    double alat,                   // lattice parameter (bohr), QE "alat"
    const std::vector<double> &zv, // zv[0..ntyp-1] (ionic charges Zval)
    const std::vector<int> &ityp,  // ityp[0..nat-1] (type index for each atom, 0-based)
    double gcutm                   // max dimensionless |G|^2 (QE style)
)
{
    const double tpi = 2.0 * M_PI;
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
    const double TWO_PI = 2.0 * M_PI;

    //the reciporacal ewald energy
    double E_rec = 0.0; 

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

                Gmax = std::max(Gmax, G); // G is already |G|^2
               // double Gmax = std::max(Gmax, 200.0 / (Lx * Lx));

                // One species type, valence charge Zval
                double tpi_over_alat = 2.0 * M_PI / alat; // bohr^-1

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

                E_rec += (4.0 * M_PI / G) * damping * std::norm(Sg); // calcutate gaussian smoothing factor for this G point
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
                        double r = std::sqrt(dx * dx + dy * dy + dz * dz); // calcualte radius for r based on points

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
        self_correction += -alpha * Z[i] * Z[i] / std::sqrt(M_PI);
    }

    double backround = 0;
    for (double Zi : Z)
    { // sum the charge nuclear charge
        backround += Zi;
    }

    backround = -M_PI * (backround * backround) / (alpha * alpha * V); // background neutrality term

    return E_real + E_rec + self_correction + backround; // sum all componets
}
double Ve_PlaneWave_3D_P(
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
    const double TWO_PI = 2.0 * M_PI;

    // compute ithe reciporacal ewald energy
    double E_rec = 0.0; // <<< CHANGED

    #pragma omp parallel for collapse(3) reduction(+ : E_rec)
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

                Gmax = std::max(Gmax, G); // G is already |G|^2
                //double Gmax = std::max(Gmax, 200.0 / (Lx * Lx));

                // One species type, valence charge Zval
                double tpi_over_alat = 2.0 * M_PI / alat; // bohr^-1

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

                E_rec += (4.0 * M_PI / G) * damping * std::norm(Sg); // calcutate gaussian smoothing factor for this G point
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
    #pragma omp parallel for collapse(3) reduction(+ : E_real)
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
        self_correction += -alpha * Z[i] * Z[i] / std::sqrt(M_PI);
    }

    double backround = 0;
    for (double Zi : Z)
    { // sum the charge nuclear charge
        backround += Zi;
    }

    backround = -M_PI * (backround * backround) / (alpha * alpha * V); // background neutrality term

    return E_real + E_rec + self_correction + backround; // sum all componets
}
int main()
{int b=64;
    // ===========================================
    // FCC Aluminum Lattice Constant (bohr)
    // ===========================================
    double a = 7.50;

    double Lx = a;
    double Ly = a;
    double Lz = a;

    // ===========================================
    // FCC primitive cell has ONE atom
    // ===========================================
    std::vector<Vec3> ion_pos;
    ion_pos.push_back({0.0, 0.0, 0.0});

    // ===========================================
    // Aluminum pseudopotential valence charge
    // ===========================================
    std::vector<double> Z = {3.0};

    // ===========================================
    // Dummy Vion_r (not used)
    // ===========================================
    std::vector<double> Vion_r;

    // Grid size — irrelevant to Ewald formula,
    // but required by your function signature.
    int Nx = 128, Ny = 128, Nz = 128;

    // ===========================================
    // Compute Ewald ion-ion energy
    // ===========================================
    double E_Ha = Ve_PlaneWave_3D_s(ion_pos, Z,
                                    Lx, Ly, Lz,
                                    Nx, Ny, Nz);

    double E_Ry = 2.0 * E_Ha;

    // ===========================================
    // Output
    // ===========================================
    std::cout << "=== FCC Aluminum Ewald Test ===\n";
    std::cout << "Lattice constant a = " << a << " bohr\n";
    std::cout << "Valence Z = 3.0\n";
    std::cout << "Primitive cell: 1 atom\n\n";

    std::cout << "Ewald ion-ion energy:\n";
    std::cout << "  " << E_Ha << " Ha\n";
    std::cout << "  " << E_Ry << " Ry\n";

    // -------------------------------------------------------
    // LATTICE CONSTANT FOR bcc IRON (bohr)
    // -------------------------------------------------------
    a = 5.421; // BCC Fe lattice constant from QE examples

    Lx = a;
    Ly = a;
    Lz = a;

    // -------------------------------------------------------
    // BCC PRIMITIVE CELL HAS 2 ATOMS:
    //   (0, 0, 0)
    //   (a/2, a/2, a/2)
    // -------------------------------------------------------

    ion_pos.push_back({0.0, 0.0, 0.0});
    ion_pos.push_back({a / 2.0, a / 2.0, a / 2.0});

    // -------------------------------------------------------
    // Fe pseudopotential valence charge from QE: Zval = 16
    // -------------------------------------------------------
    Z = {16.0, 16.0};

    // -------------------------------------------------------
    // FFT GRID — determines how many G's we sum over
    // -------------------------------------------------------
    Nx = b;
    Ny = b;
    Nz = b;

    // -------------------------------------------------------
    // Compute Ewald ion–ion energy (Hartree)
    // -------------------------------------------------------
    using namespace std::chrono;
        auto t0 = high_resolution_clock::now(); // begin timing clock
    E_Ha = Ve_PlaneWave_3D_s(ion_pos, Z, Lx, Ly, Lz, Nx, Ny, Nz);
    E_Ry = 2.0 * E_Ha;
    auto t1 = high_resolution_clock::now(); // stop clock and find total time
        std::chrono::duration<double> elapsed = t1 - t0;
    // -------------------------------------------------------
    // OUTPUT
    // -------------------------------------------------------
    std::cout << "=== BCC Iron Ewald Test ===\n";
    std::cout << "Lattice constant a = " << a << " bohr\n";
    std::cout << "Atoms:\n";
    std::cout << "  Fe at (0,0,0)\n";
    std::cout << "  Fe at (a/2,a/2,a/2)\n";
    std::cout << "Zval = 16 for both Fe atoms\n\n";

    std::cout << "Computed Ewald ion-ion energy s:\n";
    std::cout << "  " << E_Ha << " Hartree\n";
    std::cout << "  " << E_Ry << " Rydberg\n";
    std::cout << " took " << elapsed.count() << " \n";


    t0 = high_resolution_clock::now(); // begin timing clock
    E_Ha = Ve_PlaneWave_3D_P(ion_pos, Z, Lx, Ly, Lz, Nx, Ny, Nz);
    E_Ry = 2.0 * E_Ha;
  t1 = high_resolution_clock::now(); // stop clock and find total time
         elapsed = t1 - t0;
    std::cout << "Computed Ewald ion-ion energy p:\n";
    std::cout << "  " << E_Ha << " Hartree\n";
    std::cout << "  " << E_Ry << " Rydberg\n";
std::cout << " took " << elapsed.count() << " \n";
    // (QE Ewald for BCC Fe is around -693.782 Ry depending on pseudopotential)
    std::cout << "\nExpected QE magnitude ~ -693.8 Ry (depends on PP & grid)\n";
    return 0;
}
