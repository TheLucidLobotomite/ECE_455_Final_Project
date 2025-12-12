#include <iostream>
#include <vector>
#include <complex>
#include <omp.h>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <chrono>
#include <stdexcept>
// prodcution implementation fo ewald potiental code
// 3 componet vector for holding vals at a spesific pt
struct Vec3
{
    double x, y, z;
};

// calcualte alpha
// lattice_const- lattice parameter (bohr)
// zv- ionic charge for each index
// ind- index of each atom location
// g_cut- maximum abs(G)^2

double calc_alpha(double lattice_const, const std::vector<double> &zv, const std::vector<int> &ind, double g_cut)
{
    const double twopi = 2.0 * M_PI;
    const double alpha_coeff = (twopi / lattice_const) * (twopi / lattice_const); // (2π/a)^2

    double charge = 0.0;                         // total ionic charge in the unit cell
    for (int na = 0; na < (int)ind.size(); ++na) // loop through the indexs for each atom ancd calcualte the total atomic charge in the unti cell
    {
        int t = ind[na];
        charge += zv[t];
    }

    double alpha = 2.9; // start with alpha as 2.9(idk thats what the internet said)
    while (true)
    { // decreamnt alpha untill error is low enough
        alpha -= 0.1;
        if (alpha <= 0.0) // throw error if it gets to zero, alpha must be real and positve
        {
            throw std::runtime_error("bad alpha");
        }

        double arg = std::sqrt(alpha_coeff * g_cut / (4.0 * alpha));                                 // calculate ewald error bound to know how much it will damp the larges G value
        double upperbound = 2.0 * charge * charge * std::sqrt(2.0 * alpha / twopi) * std::erfc(arg); // fincd upper bound error based on arg

        if (upperbound <= 1.0e-7) // if the upper error bound is within threshold break
            break;
    }

    return alpha; // return alpha val
}

// convert fractional cyrastal points to cartesian for cubic
static inline Vec3 reciprocal_cart_cubic(const Vec3 &tau_cryst, double lattice_const)
{
    Vec3 r;
    r.x = tau_cryst.x * lattice_const;
    r.y = tau_cryst.y * lattice_const;
    r.z = tau_cryst.z * lattice_const;
    return r;
}

// lattice_const- latice constant distance a
//  atom positions in crystal fractional coordinates
//  list of ionic valence charges from the pseudopotentials
//  g_cut- cutoff g-val- for deterimining error
// E_real- real space energy vector
//  E_rec- recipracal space energy vector
//  E_self- self correction energy vector
//  E_backround- backround correctiosn energy vector
double V_ewald_s(double lattice_const, const std::vector<Vec3> &tau_cryst, const std::vector<double> &Zatom, double g_cut, double &E_real, double &E_rec, double &E_self, double &E_back)
{
    const int nat = (int)tau_cryst.size();

    // build zv and ind arrays for alpha choice
    std::vector<double> zv(1);    // one species
    zv[0] = Zatom[0];             // all atoms same Z
    std::vector<int> ind(nat, 0); // all atoms type 0

    // choose alpha as QE does
    double alpha = calc_alpha(lattice_const, zv, ind, g_cut); // alpha used in Ewald sum
    double eta = std::sqrt(alpha);                            // eta = sqrt(alpha)

    // cell volume and reciprocal prefactors for cubic cell
    const double V = lattice_const * lattice_const * lattice_const; // cell volume
    const double twopiba = 2.0 * M_PI / lattice_const;
    const double alpha_coeff = twopiba * twopiba;

    // convert crystal positions to cartesian
    std::vector<Vec3> r_cart(nat);
    for (int i = 0; i < nat; ++i)
    {
        r_cart[i] = reciprocal_cart_cubic(tau_cryst[i], lattice_const); // cartesian in bohr
    }
    // above here
    //  calcualte realspace ewald energy
    E_real = 0.0;

    double Rcut = 4.0 / std::sqrt(alpha);      // real-space cutoff in bohr
    int NxR = (int)(Rcut / lattice_const) + 1; // number of image cells in each direction
    int NyR = NxR;
    int NzR = NxR;
    // loop through each of the periodic cells by looping through each dim
    for (int nx = -NxR; nx <= NxR; ++nx)         // x-loop
        for (int ny = -NyR; ny <= NyR; ++ny)     // y-loop
            for (int nz = -NzR; nz <= NzR; ++nz) // z-loop
            {
                double Rx = nx * lattice_const; // compute the translation vectors, ie the distance to shift the ions postion into the next cell
                double Ry = ny * lattice_const;
                double Rz = nz * lattice_const;

                for (int i = 0; i < nat; ++i) // loop through each pair of ions
                    for (int j = 0; j < nat; ++j)
                    {
                        if (nx == 0 && ny == 0 && nz == 0 && i == j) // ignore self interactions for ions
                            continue;

                        double dx = r_cart[i].x - r_cart[j].x + Rx;
                        double dy = r_cart[i].y - r_cart[j].y + Ry;
                        double dz = r_cart[i].z - r_cart[j].z + Rz;

                        double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                        if (r == 0.0)
                            continue;
                        if (r > Rcut)
                            continue; // ensure that the ion is in the radius of cutoff

                        E_real += Zatom[i] * Zatom[j] * std::erfc(eta * r) / r; // using the shortrange erfc aproximation to remove long range interactions
                    }
            }

    E_real *= 0.5; // correct for double sum

    // the reciporacal ewald energy
    E_rec = 0.0;

    // dimensionless G^2 is (|G|^2 / alpha_coeff) = h^2 + k^2 + l^2 for cubic cell
    int maxh = (int)std::ceil(std::sqrt(g_cut)) + 1; // max index in each direction

    for (int h = -maxh; h <= maxh; ++h)
    {

        for (int k = -maxh; k <= maxh; ++k)
        {
            for (int l = -maxh; l <= maxh; ++l)
            {
                if (h == 0 && k == 0 && l == 0)
                    continue; // skip G=0

                double G_dimless = double(h * h + k * k + l * l); // (abs(G)^2 / alpha_coeff)
                if (G_dimless > g_cut)
                    continue; // outside G cutoff

                double Gx = twopiba * h; // reciprocal vector components
                double Gy = twopiba * k;
                double Gz = twopiba * l;

                double G = Gx * Gx + Gy * Gy + Gz * Gz; // abs(G)^2 in bohr^-2

                // compute structure factor S(G) 
                std::complex<double> Sg(0.0, 0.0);
                for (int i = 0; i < nat; ++i)
                {
                    double dot = Gx * r_cart[i].x + Gy * r_cart[i].y + Gz * r_cart[i].z; // G·r
                    double c = std::cos(dot);
                    double s = std::sin(dot);
                    std::complex<double> phase(c, s); // e^{i G·r}
                    Sg += Zatom[i] * phase;
                }

                double damping = std::exp(-G / (4.0 * alpha)); // Gaussian screening exp(-G^2/(4α))
                double SG = std::norm(Sg);                     // abs(S(G))^2

                E_rec += (4.0 * M_PI / G) * damping * SG; // reciprocal contribution
            }
        }
    }

    E_rec *= (1.0 / (2.0 * V)); // prefactor 1/(2V) for energy
                                // below here
    //  calcualte the self correting term
    E_self = 0.0;
    for (int i = 0; i < nat; ++i) // remove self interaction of each charge
    {
        E_self += -eta * Zatom[i] * Zatom[i] / std::sqrt(M_PI);
    }

    double qtot = 0.0;
    for (int i = 0; i < nat; ++i) // total ionic charge
        qtot += Zatom[i];

    E_back = -M_PI * (qtot * qtot) / (alpha * V); // neutralizing background term

    // sum all componets
    double Etotal = E_real + E_rec + E_self + E_back;
    return Etotal;
}

double V_ewald_p(double lattice_const, const std::vector<Vec3> &tau_cryst, const std::vector<double> &Zatom, double g_cut, double &E_real, double &E_rec, double &E_self, double &E_back)
{
    const int nat = (int)tau_cryst.size();

    // build zv and ind arrays for alpha choice
    std::vector<double> zv(1);    // one species
    zv[0] = Zatom[0];             // all atoms same Z
    std::vector<int> ind(nat, 0); // all atoms type 0

    // choose alpha as QE does
    double alpha = calc_alpha(lattice_const, zv, ind, g_cut); // alpha used in Ewald sum
    double eta = std::sqrt(alpha);                            // eta = sqrt(alpha)

    // cell volume and reciprocal prefactors for cubic cell
    const double V = lattice_const * lattice_const * lattice_const; // cell volume
    const double twopiba = 2.0 * M_PI / lattice_const;
    const double alpha_coeff = twopiba * twopiba;

    // convert crystal positions to cartesian
    std::vector<Vec3> r_cart(nat);
    for (int i = 0; i < nat; ++i)
    {
        r_cart[i] = reciprocal_cart_cubic(tau_cryst[i], lattice_const); // cartesian in bohr
    }
    // above here
    //  calcualte realspace ewald energy
    E_real = 0.0;

    double Rcut = 4.0 / std::sqrt(alpha);      // real-space cutoff in bohr
    int NxR = (int)(Rcut / lattice_const) + 1; // number of image cells in each direction
    int NyR = NxR;
    int NzR = NxR;
// loop through each of the periodic cells by looping through each dim
#pragma omp parallel for collapse(3) reduction(+ : E_real)
    for (int nx = -NxR; nx <= NxR; ++nx)         // x-loop
        for (int ny = -NyR; ny <= NyR; ++ny)     // y-loop
            for (int nz = -NzR; nz <= NzR; ++nz) // z-loop
            {
                double Rx = nx * lattice_const; // compute the translation vectors, ie the distance to shift the ions postion into the next cell
                double Ry = ny * lattice_const;
                double Rz = nz * lattice_const;

                for (int i = 0; i < nat; ++i) // loop through each pair of ions
                    for (int j = 0; j < nat; ++j)
                    {
                        if (nx == 0 && ny == 0 && nz == 0 && i == j) // ignore self interactions for ions
                            continue;

                        double dx = r_cart[i].x - r_cart[j].x + Rx;
                        double dy = r_cart[i].y - r_cart[j].y + Ry;
                        double dz = r_cart[i].z - r_cart[j].z + Rz;

                        double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                        if (r == 0.0)
                            continue;
                        if (r > Rcut)
                            continue; // ensure that the ion is in the radius of cutoff

                        E_real += Zatom[i] * Zatom[j] * std::erfc(eta * r) / r; // using the shortrange erfc aproximation to remove long range interactions
                    }
            }

    E_real *= 0.5; // correct for double sum

    //reciporacal ewald energy
    E_rec = 0.0;

    // dimensoinless G^2 is (abs(G)^2 / alpha_coeff) = h^2 + k^2 + l^2 for cubic cell
    int maxh = (int)std::ceil(std::sqrt(g_cut)) + 1; // max index in each direction
#pragma omp parallel for collapse(3) reduction(+ : E_rec)
    for (int h = -maxh; h <= maxh; ++h)
    {

        for (int k = -maxh; k <= maxh; ++k)
        {
            for (int l = -maxh; l <= maxh; ++l)
            {
                if (h == 0 && k == 0 && l == 0)
                    continue; // skip G=0

                double G_dimless = double(h * h + k * k + l * l); // (abs(G)^2 / alpha_coeff)
                if (G_dimless > g_cut)
                    continue; // ensure  pt is outside G cutoff

                double Gx = twopiba * h; // calcualte recipraocal vector componnents
                double Gy = twopiba * k;
                double Gz = twopiba * l;

                double G = Gx * Gx + Gy * Gy + Gz * Gz; // calcualte g magintude as abs(G)^2 in bohr^-2

                // calculate structure factor S(G) = sum( Z e^(i G*r))
                std::complex<double> Sg(0.0, 0.0);
                for (int i = 0; i < nat; ++i)
                {
                    double dot = Gx * r_cart[i].x + Gy * r_cart[i].y + Gz * r_cart[i].z; // G*r
                    double c = std::cos(dot);
                    double s = std::sin(dot);
                    std::complex<double> phase(c, s); // e^(i G*r)
                    Sg += Zatom[i] * phase;
                }

                double damping = std::exp(-G / (4.0 * alpha)); // apply gaussian screening exp(-G^2/(4aplha)) ?
                double SG = std::norm(Sg);                     // normalizze strucutre factor 

                E_rec += (4.0 * M_PI / G) * damping * SG; // summ reciprocal engery 
            }
        }
    }

    E_rec *= (1.0 / (2.0 * V)); // compute reciporacl engergy  by 1/(2V) for energy to normalize
                                // below here
    //  calcualte the self correting term
    E_self = 0.0;
    for (int i = 0; i < nat; ++i) // remove self interaction of each charge
    {
        E_self += -eta * Zatom[i] * Zatom[i] / std::sqrt(M_PI);
    }

    double qtot = 0.0;
    for (int i = 0; i < nat; ++i) // calculatre the total ionic charge
        qtot += Zatom[i];

    E_back = -M_PI * (qtot * qtot) / (alpha * V); // calcualte the  background corection term

    // sum all componets
    double Etotal = E_real + E_rec + E_self + E_back;
    return Etotal;
}
