#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <cmath>
#include <chrono>

#define PI 3.14159265358979323846

struct UPF {
    std::string element;
    double z_valence;
    bool ultrasoft = false;
    bool paw = false;
    int lmax = -1;
    int mesh = 0;
    int nproj = 0;

    std::vector<double> r, rab;
    std::vector<double> vloc;
    std::vector<std::vector<double>> beta;
    std::vector<int> beta_l;
    std::vector<std::vector<double>> D;
    std::vector<std::vector<double>> Q;
};

// trims all the garbage off of the string
std::string trim_string(const std::string& str) {
    size_t start = str.find_first_not_of(" \n\r\t");
    size_t end = str.find_last_not_of(" \n\r\t");
    return (start == std::string::npos) ? "" : str.substr(start, end - start + 1);
}

std::vector<double> read_data_for_tag(std::istream& file, const std::string& tag){
    std::vector<double> data;
    std::string line;
    while (std::getline(file, line)){
        line = trim_string(line);

        if (line.find("</" + tag) != std::string::npos) break;
        if (line.find("<" + tag) != std::string::npos) continue;

        std::stringstream ss(line);
        double val;
        while (ss >> val) data.push_back(val);
    }
    return data;
}

UPF read_upf(const std::string& filename) {
    UPF upf;
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;
    bool mesh = false, nonlocal = false, aug = false, in_header = false;

    while (std::getline(file, line)){
        line = trim_string(line);

        // PP_HEADER can span multiple lines - read until we hit "/>"
        if (line.find("<PP_HEADER") != std::string::npos) {
            in_header = true;
        }

        if (in_header) {
            // element
            size_t pos = line.find("element=");
            if (pos != std::string::npos) {
                upf.element = line.substr(pos + 9, 2);
            }
            // z_valence - fixed to handle scientific notation
            size_t zpos = line.find("z_valence=\"");
            if (zpos != std::string::npos){
                size_t start = zpos + 11;
                size_t end = line.find("\"", start);
                std::string zval_str = line.substr(start, end - start);
                upf.z_valence = std::stod(zval_str);
            }
            // ultrasoft
            if (line.find("is_ultrasoft=\"T\"") != std::string::npos || line.find("is_ultrasoft=\"true\"") != std::string::npos){
                upf.ultrasoft = true;
            }
            // PAW
            if (line.find("is_paw=\"T\"") != std::string::npos || line.find("is_paw=\"true\"") != std::string::npos){
                upf.paw = true;
            }
            // mesh_size - fixed to handle any number of digits
            size_t meshpos = line.find("mesh_size=\"");
            if (meshpos != std::string::npos){
                size_t start = meshpos + 11;
                size_t end = line.find("\"", start);
                upf.mesh = std::stoi(line.substr(start, end - start));
            }
            // number_of_proj - fixed to handle any number of digits
            size_t pos2 = line.find("number_of_proj=\"");
            if (pos2 != std::string::npos){
                size_t start = pos2 + 16;
                size_t end = line.find("\"", start);
                upf.nproj = std::stoi(line.substr(start, end - start));
            }
            // l_max
            size_t lmaxpos = line.find("l_max=\"");
            if (lmaxpos != std::string::npos){
                size_t start = lmaxpos + 7;
                size_t end = line.find("\"", start);
                upf.lmax = std::stoi(line.substr(start, end - start));
            }

            // End of header tag
            if (line.find("/>") != std::string::npos) {
                in_header = false;
            }
        }

        if (line.find("<PP_MESH") != std::string::npos) mesh = true;
        if (line.find("</PP_MESH>") != std::string::npos) mesh = false;

        if (line.find("<PP_NONLOCAL") != std::string::npos) nonlocal = true;
        if (line.find("</PP_NONLOCAL>") != std::string::npos) nonlocal = false;

        if (line.find("<PP_AUGMENTATION") != std::string::npos) aug = true;
        if (line.find("</PP_AUGMENTATION>") != std::string::npos) aug = false;

        // radial grid
        if (mesh) {
            if (line.find("<PP_R") != std::string::npos && line.find("<PP_RAB") == std::string::npos) {
                upf.r = read_data_for_tag(file, "PP_R");
            }
            if (line.find("<PP_RAB") != std::string::npos){
                upf.rab = read_data_for_tag(file, "PP_RAB");
            }
        }

        if (line.find("<PP_LOCAL") != std::string::npos){
            upf.vloc = read_data_for_tag(file, "PP_LOCAL");
        }

        // projectors
        if (nonlocal && line.find("<PP_BETA.") != std::string::npos) {
            int n, l = 0;
            size_t pos3 = line.find("<PP_BETA.");
            size_t pos4 = line.find("angular_momentum=");
            int betaindex = std::stoi(line.substr(pos3 + 9, 1)) - 1;
            int angularmomentum = std::stoi(line.substr(pos4 + 18, 1));
            std::vector<double> rawbeta = read_data_for_tag(file, "PP_BETA." + std::to_string(betaindex + 1));
            // UPF format stores β(r), not β(r)/r, so use it directly
            upf.beta.push_back(rawbeta);
            upf.beta_l.push_back(angularmomentum);
        }

        // D_IJ matrix
        if (nonlocal && line.find("<PP_DIJ") != std::string::npos) {
            std::vector<double> dij = read_data_for_tag(file, "PP_DIJ");
            int n = upf.nproj;
            upf.D.assign(n, std::vector<double>(n, 0.0));
            int k = 0;
            for (int i = 0; i < n; ++i){
                for (int j = 0; j < n; ++j){
                    if (k < (int)dij.size()){
                        upf.D[i][j] = upf.D[j][i] = dij[k];
                        k++;
                    }
                }
            }
        }

        // Q matrix
        if (nonlocal && line.find("<PP_Q") != std::string::npos) {
            std::vector<double> q = read_data_for_tag(file, "PP_Q");
            int n = upf.nproj;
            upf.Q.assign(n, std::vector<double>(n, 0.0));
            int k = 0;
            for (int i = 0; i < n; ++i){
                for (int j = 0; j < n; ++j){
                    if (k < (int)q.size()){
                        upf.Q[i][j] = upf.Q[j][i] = q[k];
                        k++;
                    }
                }
            }
        }
    }

    return upf;
}

/**
 * Spherical Bessel function j_l(x)
 */
inline double spherical_bessel(int l, double x) {
    if (x < 1e-8) {
        // Use small-x expansion
        if (l == 0) return 1.0;
        if (l == 1) return x / 3.0;
        if (l == 2) return x*x / 15.0;
        return 0.0;
    }

    if (l == 0) return sin(x) / x;
    if (l == 1) return (sin(x) / (x*x)) - (cos(x) / x);
    if (l == 2) return ((3.0/(x*x) - 1.0) * sin(x) / x) - (3.0 * cos(x) / (x*x));
    return 0.0;
}

/**
 * Lookup table for beta projectors in reciprocal space
 */
struct BetaCache {
    std::vector<std::vector<double>> beta_q_table;  // [projector_index][q_index]
    double q_max;
    double dq;
    int nq;

    // Linear interpolation
    double get_beta_q(int iproj, double q) const {
        if (q >= q_max) return 0.0;
        double idx_f = q / dq;
        int idx = (int)idx_f;
        if (idx >= nq - 1) return beta_q_table[iproj][nq - 1];

        double frac = idx_f - idx;
        return beta_q_table[iproj][idx] * (1.0 - frac) + beta_q_table[iproj][idx + 1] * frac;
    }
};

/**
 * Pre-compute beta projector Fourier transforms on a grid
 */
BetaCache precompute_beta_cache(UPF& upf, double q_max) {
    BetaCache cache;
    cache.q_max = q_max * 1.5;  // Add safety margin
    cache.nq = 2000;  // Fine grid for accuracy
    cache.dq = cache.q_max / cache.nq;

    std::cout << "  Pre-computing beta projector cache (0 to " << cache.q_max << " bohr^-1)...\n";

    cache.beta_q_table.resize(upf.nproj);

    for (int iproj = 0; iproj < upf.nproj; ++iproj) {
        int l = upf.beta_l[iproj];
        cache.beta_q_table[iproj].resize(cache.nq);

        for (int iq = 0; iq < cache.nq; ++iq) {
            double q = iq * cache.dq;

            // Compute β̃_l(q) = 4π ∫ β_l(r) j_l(qr) r² dr
            double beta_q = 0.0;
            for (int ir = 0; ir < upf.r.size(); ++ir) {
                if (upf.r[ir] < 1e-12) continue;
                double r = upf.r[ir];
                double x = q * r;
                double jl = spherical_bessel(l, x);
                beta_q += upf.beta[iproj][ir] * jl * r*r * upf.rab[ir];
            }
            beta_q *= 4.0 * PI;

            cache.beta_q_table[iproj][iq] = beta_q;
        }
    }

    std::cout << "  Beta cache computed (" << cache.nq << " q-points per projector)\n";
    return cache;
}

/**
 * Compute pseudopotential matrix element using cached beta values
 * V_nl(G,G') = Σ_ij D_ij * β̃_i(q) * β̃_j(q) where q = |G - G'|
 *
 * NOTE: Each beta includes 4π, so β̃_i * β̃_j includes (4π)².
 * However, the Kleinman-Bylander formula should only have 4π once.
 * We divide by 4π to correct this.
 */
double Vnl_GGp_cached(UPF& upf, const BetaCache& cache, const GVector& g1, const GVector& g2) {
    // Compute |G - G'| from the vector components
    double dx = g1.gx - g2.gx;
    double dy = g1.gy - g2.gy;
    double dz = g1.gz - g2.gz;
    double q = std::sqrt(dx*dx + dy*dy + dz*dz);  // |G - G'| in bohr⁻¹

    double V_total = 0.0;

    // Sum over all projectors: V_nl = Σ_ij D_ij * β̃_i(q) * β̃_j(q)
    for (int i = 0; i < upf.nproj; ++i) {
        double beta_i_q = cache.get_beta_q(i, q);

        for (int j = 0; j < upf.nproj; ++j) {
            double beta_j_q = cache.get_beta_q(j, q);

            // Add contribution: D_ij * β̃_i * β̃_j
            // D is in Hartree, convert to Rydberg (*2)
            // Divide by 4π to avoid (4π)² from squaring beta factors
            V_total += 2.0 * upf.D[i][j] * beta_i_q * beta_j_q / (4.0 * PI);
        }
    }

    return V_total;
}

/**
 * Pre-compute the full pseudopotential matrix (DENSITY INDEPENDENT!)
 * This should be called ONCE before the SCF loop
 *
 * @param upf Pseudopotential structure
 * @param gvectors Vector of G-vectors
 * @param npw Number of plane waves
 * @param cell_volume Unit cell volume in Bohr³
 * @return Pre-computed Vnl matrix (npw x npw)
 */
double** precompute_pseudopotential_matrix(UPF& upf, const std::vector<GVector>& gvectors, int npw, double cell_volume) {
    std::cout << "  Pre-computing pseudopotential matrix (" << npw << "x" << npw << ")...\n";

    // DIAGNOSTIC: Print D[0][0] value
    std::cout << "\n  DIAGNOSTIC INFO:\n";
    std::cout << "  D[0][0] = " << upf.D[0][0] << " (should be ~0.1-1.0 in Hartree)\n";
    std::cout << "  D[0][0] * 2 = " << (2.0 * upf.D[0][0]) << " (Rydberg conversion)\n\n";

    // Find maximum |G-G'| for cache range
    double q_max = 0.0;
    for (int i = 0; i < npw; i++) {
        for (int j = 0; j < npw; j++) {
            double dx = gvectors[i].gx - gvectors[j].gx;
            double dy = gvectors[i].gy - gvectors[j].gy;
            double dz = gvectors[i].gz - gvectors[j].gz;
            double q = std::sqrt(dx*dx + dy*dy + dz*dz);
            q_max = std::max(q_max, q);
        }
        if (i > 100) break;  // Sample first 100 is enough to estimate
    }
    q_max *= 1.2;  // Add safety margin

    // Pre-compute beta cache (MUCH faster than computing on-the-fly!)
    BetaCache cache = precompute_beta_cache(upf, q_max);

    auto t_start = std::chrono::high_resolution_clock::now();

    // Allocate matrix
    double** Vnl = new double*[npw];
    for (int i = 0; i < npw; i++) {
        Vnl[i] = new double[npw];
    }

    // Track min/max/avg for diagnostics
    double vnl_min = 1e99, vnl_max = -1e99, vnl_sum = 0.0;
    int count = 0;

    // Compute all matrix elements (exploit symmetry: only compute upper triangle)
    // Normalize by cell volume (plane wave basis normalization: 1/√Ω)
    double volume_norm = 1.0 / cell_volume;

    for (int i = 0; i < npw; i++) {
        for (int j = i; j < npw; j++) {  // Start from j=i (upper triangle only)
            double val = Vnl_GGp_cached(upf, cache, gvectors[i], gvectors[j]) * volume_norm;
            Vnl[i][j] = val;
            Vnl[j][i] = val;  // Exploit symmetry

            // Track statistics
            vnl_min = std::min(vnl_min, val);
            vnl_max = std::max(vnl_max, val);
            vnl_sum += val;
            count++;
            if (i != j) {
                vnl_sum += val;  // Count the symmetric element too
                count++;
            }
        }

        // Progress indicator every 5%
        if ((i + 1) % (npw / 20 + 1) == 0) {
            std::cout << "    Progress: " << (100 * (i + 1) / npw) << "%\n";
        }
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double time_pseudo = std::chrono::duration<double>(t_end - t_start).count();
    
    std::cout << "  Pseudopotential matrix computed in " << std::fixed << std::setprecision(2) 
              << time_pseudo << " s\n";
    std::cout << "  (This is done ONCE - will be reused in all SCF iterations)\n\n";
    
    // Print diagnostic statistics
    std::cout << "  PSEUDOPOTENTIAL MATRIX STATISTICS:\n";
    std::cout << "  Min value: " << std::scientific << std::setprecision(3) << vnl_min << " Ry\n";
    std::cout << "  Max value: " << std::scientific << std::setprecision(3) << vnl_max << " Ry\n";
    std::cout << "  Avg value: " << std::scientific << std::setprecision(3) << (vnl_sum / count) << " Ry\n";
    std::cout << "  (These should be O(1-10) Ry, not O(1000+) Ry)\n\n";
    
    return Vnl;
}

/**
 * Compute local pseudopotential in reciprocal space
 * V_loc(G) = 4π/Ω ∫ V_loc(r) j₀(Gr) r² dr
 *
 * For spherically symmetric systems, this is the Fourier-Bessel transform.
 *
 * @param upf Pseudopotential structure
 * @param gvectors Vector of G-vectors
 * @param npw Number of plane waves
 * @param cell_volume Unit cell volume in Bohr³
 * @return V_loc values for each G-vector (in Rydberg)
 */
std::vector<double> compute_vloc_g(UPF& upf, const std::vector<GVector>& gvectors, int npw, double cell_volume) {
    std::cout << "  Computing local pseudopotential V_loc(G)...\n";

    std::vector<double> vloc_g(npw);

    for (int ig = 0; ig < npw; ig++) {
        double G = std::sqrt(gvectors[ig].g2);

        if (G < 1e-8) {
            // G=0 case: Set to zero to avoid divergence
            // The Coulomb tail -Z/r contribution at G=0 is handled by Ewald sum
            vloc_g[ig] = 0.0;
        } else {
            // Compute Fourier-Bessel transform: ∫ V_loc(r) j₀(Gr) r² dr
            double vloc_ft = 0.0;
            for (int ir = 0; ir < upf.r.size(); ++ir) {
                if (upf.r[ir] < 1e-12) continue;
                double r = upf.r[ir];
                double x = G * r;
                double j0 = sin(x) / x;  // j₀(x) = sin(x)/x

                // Subtract the Coulomb tail: -2*Z_v/r (factor of 2 for Rydberg)
                // This removes the long-range part that's already in Ewald
                double vloc_short = upf.vloc[ir] + 2.0 * upf.z_valence / r;

                vloc_ft += vloc_short * j0 * r*r * upf.rab[ir];
            }

            // V_loc(G) = 4π/Ω × integral
            vloc_g[ig] = 4.0 * PI * vloc_ft / cell_volume;
        }
    }

    std::cout << "  V_loc(G=0) = " << std::fixed << std::setprecision(6) << vloc_g[0] << " Ry (set to zero)\n";
    std::cout << "  V_loc computed for " << npw << " G-vectors\n\n";

    return vloc_g;
}

/**
 * Free pseudopotential matrix
 */
void free_pseudopotential_matrix(double** Vnl, int npw) {
    if (Vnl) {
        for (int i = 0; i < npw; i++) {
            delete[] Vnl[i];
        }
        delete[] Vnl;
    }
}


