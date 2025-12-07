#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <stdexcept>

struct Vec3
{
    double x, y, z;
};
// lattice parameter (bohr), QE "alat"
// zv[0..ntyp-1]  (ionic charges Zval)
// ityp[0..nat-1] (type index for each atom, 0-based)
// max dimensionless |G|^2 (QE style)

double compute_alpha_qe( double alat, const std::vector<double> &zv, const std::vector<int> &ityp,  double gcutm  )                
{
    const double tpi = 2.0 * M_PI;
    const double tpiba2 = (tpi / alat) * (tpi / alat); // (2π/a)^2

    double charge = 0.0; // total ionic charge in the unit cell
    for (int na = 0; na < (int)ityp.size(); ++na)
    {
        int t = ityp[na]; // type index
        charge += zv[t];
    }

    double alpha = 2.9; // start from alpha = 2.9 and decrease by 0.1 until upperbound < 1e-7
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

double compute_rmax_realspace_qe(double alpha)
{
    // rr such that sqrt(alpha)*rr = 4 -> rr = 4 / sqrt(alpha)
    return 4.0 / std::sqrt(alpha); // in bohr
}

// convert crystal coord (fractional) to cartesian for cubic ibrav=1
static inline Vec3 cryst_to_cart_cubic(const Vec3 &tau_cryst, double alat)
{
    Vec3 r;
    r.x = tau_cryst.x * alat;
    r.y = tau_cryst.y * alat;
    r.z = tau_cryst.z * alat;
    return r;
}
// cell parameter (bohr)
// Z[i] for each atom
// dimensionless |G|^2 cutoff (QE-style)
double V_ewald(double alat, const std::vector<Vec3> &tau_cryst, const std::vector<double> &Zatom,    double gcutm,  double &E_real, double &E_rec, double &E_self, double &E_back)
{
    const int nat = (int)tau_cryst.size();

    // build zv and ityp arrays for alpha choice
    std::vector<double> zv(1); // one species
    zv[0] = Zatom[0];          // all atoms same Z
    std::vector<int> ityp(nat, 0); // all atoms type 0

    // choose alpha as QE does
    double alpha = compute_alpha_qe(alat, zv, ityp, gcutm); // alpha used in Ewald sum
    double eta = std::sqrt(alpha);                          // eta = sqrt(alpha)

    // cell volume and reciprocal prefactors for cubic cell
    const double V = alat * alat * alat; // cell volume
    const double tpiba = 2.0 * M_PI / alat;
    const double tpiba2 = tpiba * tpiba;

    // convert crystal positions to cartesian
    std::vector<Vec3> r_cart(nat);
    for (int i = 0; i < nat; ++i)
    {
        r_cart[i] = cryst_to_cart_cubic(tau_cryst[i], alat); // cartesian in bohr
    }
//above here
   // calcualte realspace ewald energy
    E_real = 0.0;

    double Rcut = compute_rmax_realspace_qe(alpha); // real-space cutoff in bohr
    int NxR = (int)(Rcut / alat) + 1;               // number of image cells in each direction
    int NyR = NxR;
    int NzR = NxR;
// loop through each of the periodic cells by looping through each dim
    for (int nx = -NxR; nx <= NxR; ++nx) // x-loop
        for (int ny = -NyR; ny <= NyR; ++ny)// y-loop 
            for (int nz = -NzR; nz <= NzR; ++nz)// z-loop 
            {
                double Rx = nx * alat;  // compute the translation vectors, ie the distance to shift the ions postion into the next cell
                double Ry = ny * alat;
                double Rz = nz * alat;

                for (int i = 0; i < nat; ++i)// loop through each pair of ions
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

 
    //the reciporacal ewald energy
    E_rec = 0.0;

    // dimensionless G^2 is (|G|^2 / tpiba2) = h^2 + k^2 + l^2 for cubic cell
    int maxh = (int)std::ceil(std::sqrt(gcutm)) + 1; // max index in each direction

    for (int h = -maxh; h <= maxh; ++h){

        for (int k = -maxh; k <= maxh; ++k)
        {
            for (int l = -maxh; l <= maxh; ++l)
            {
                if (h == 0 && k == 0 && l == 0)
                    continue; // skip G=0

                double G_dimless = double(h * h + k * k + l * l); // (|G|^2 / tpiba2)
                if (G_dimless > gcutm)
                    continue; // outside G cutoff

                double Gx = tpiba * h; // reciprocal vector components
                double Gy = tpiba * k;
                double Gz = tpiba * l;

                double G = Gx * Gx + Gy * Gy + Gz * Gz; // |G|^2 in bohr^-2

                // compute structure factor S(G) = Σ_i Z_i e^{i G·r_i}
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
                double SG = std::norm(Sg);                     // |S(G)|^2

                E_rec += (4.0 * M_PI / G) * damping * SG; // reciprocal contribution
            }
        }
    }

    E_rec *= (1.0 / (2.0 * V)); // prefactor 1/(2V) for energy
//below here
// calcualte the self correting term
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

// very simple parser for your iron.in (ibrav=1, cubic, ATOMIC_POSITIONS crystal)
bool read_iron_in(
    const std::string &filename,
    double &alat,
    double &ecutrho,
    int &nat,
    std::vector<Vec3> &tau_cryst)
{
    std::ifstream in(filename);
    if (!in)
        return false;

    std::string line;
    bool in_positions = false;
    nat = 0;
    alat = 0.0;
    ecutrho = 0.0;

    while (std::getline(in, line))
    {
        std::string trimmed = line;
        // crude trim
        while (!trimmed.empty() && (trimmed.back() == ' ' || trimmed.back() == '\t' || trimmed.back() == '\r'))
            trimmed.pop_back();

        if (trimmed.find("celldm(1)=") != std::string::npos)
        {
            // example:  ibrav= 1, nat= 4, ntyp= 1, celldm(1)=6.767109,
            std::size_t p = trimmed.find("celldm(1)=");
            if (p != std::string::npos)
            {
                p += std::string("celldm(1)=").size();
                std::stringstream ss(trimmed.substr(p));
                ss >> alat;
            }
        }

        if (trimmed.find("nat=") != std::string::npos)
        {
            std::size_t p = trimmed.find("nat=");
            if (p != std::string::npos)
            {
                p += std::string("nat=").size();
                std::stringstream ss(trimmed.substr(p));
                ss >> nat;
            }
        }

        if (trimmed.find("ecutrho") != std::string::npos)
        {
            // example: ecutwfc = 50, ecutrho = 500,
            std::size_t p = trimmed.find("ecutrho");
            if (p != std::string::npos)
            {
                p = trimmed.find('=', p);
                if (p != std::string::npos)
                {
                    ++p;
                    std::stringstream ss(trimmed.substr(p));
                    ss >> ecutrho;
                }
            }
        }

        if (trimmed.find("ATOMIC_POSITIONS") != std::string::npos)
        {
            in_positions = true;
            continue;
        }

        if (in_positions)
        {
            if (trimmed.empty() || trimmed.find("K_POINTS") != std::string::npos)
            {
                in_positions = false;
                continue;
            }

            std::stringstream ss(trimmed);
            std::string label;
            Vec3 tau;
            ss >> label >> tau.x >> tau.y >> tau.z;
            tau_cryst.push_back(tau);
        }
    }

    if (nat == 0 || alat == 0.0 || tau_cryst.size() == 0)
        return false;

    return true;
}

int main()
{
    // read QE iron.in file
    double alat, ecutrho;
    int nat;
    std::vector<Vec3> tau_cryst;

    if (!read_iron_in("iron.in", alat, ecutrho, nat, tau_cryst))
    {
        std::cerr << "Error: could not parse iron.in\n";
        return 1;
    }

    // assume one species, all atoms same valence Zval
    // set this to match your Fe pseudopotential Zval (for example 16.0)
    double Zval = 16.0;

    std::vector<double> Zatom(nat, Zval); // ionic charges for each atom

    // approximate gcutm from ecutrho and (2π/a)^2
    // QE defines dimensionless G^2 = |G|^2 / tpiba2, gcutm is the max of this.
    const double tpiba = 2.0 * M_PI / alat;
    const double tpiba2 = tpiba * tpiba;
    // this mapping is approximate; you can tune the prefactor to match QE exactly
    double gcutm = ecutrho / tpiba2; // simple guess

    double E_real, E_rec, E_self, E_back;
    double E_Ha = V_ewald(alat, tau_cryst, Zatom, gcutm,
                                  E_real, E_rec, E_self, E_back);
    double E_Ry = 2.0 * E_Ha;

    std::cout << std::setprecision(10);
    std::cout << "=== QE-style Ewald test from iron.in ===\n";
    std::cout << "alat      = " << alat << " bohr\n";
    std::cout << "nat       = " << nat << "\n";
    std::cout << "ecutrho   = " << ecutrho << " Ry\n";
    std::cout << "Zval      = " << Zval << "\n";
    std::cout << "gcutm     = " << gcutm << "\n\n";

    std::cout << "E_real    = " << E_real << " Ha\n";
    std::cout << "E_rec     = " << E_rec  << " Ha\n";
    std::cout << "E_self    = " << E_self << " Ha\n";
    std::cout << "E_back    = " << E_back << " Ha\n";
    std::cout << "----------------------------------------\n";
    std::cout << "E_ewald   = " << E_Ha  << " Ha\n";
    std::cout << "E_ewald   = " << E_Ry  << " Ry\n";


    return 0;
}
