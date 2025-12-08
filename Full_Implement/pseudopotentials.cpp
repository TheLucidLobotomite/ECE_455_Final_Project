#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdio.h>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <map>
#include <set>
#include <complex>

#define PI 3.14159265358979323846

struct UPF {
    std::string element;
    double z_valence;
    bool ultrasoft = false;
    bool paw = false;
    int lmax = -1;
    int mesh = 0;
    int nproj = 0;
    int nqf = 0;  // number of Q functions

    std::vector<double> r, rab;
    std::vector<double> vloc;
    std::vector<std::vector<double>> beta;
    std::vector<int> beta_l;
    std::vector<std::vector<double>> D;
    std::vector<std::vector<double>> Q;  // scalar Q_IJ integrals

    // For ultrasoft: Q_IJ(r) augmentation functions
    // qfunc[i][j] is a map from l -> radial function Q_IJ^l(r)
    std::vector<std::vector<std::map<int, std::vector<double>>>> qfunc;  // qfunc[i][j][l] = Q_IJ^l(r)

    // Cached beta form factors β_i(q) = ∫ β_i(r) j_l(qr) r² dr
    // Computed once and reused for both V_nl and augmentation charge
    std::map<double, std::vector<double>> beta_q_cache;
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
    bool found_closing_tag = false;
    while (std::getline(file, line)){
        line = trim_string(line);

        if (line.find("</" + tag) != std::string::npos) {
            found_closing_tag = true;
            break;
        }
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
    bool mesh = false, nonlocal = false, aug = false;

    while (std::getline(file, line)){
        line = trim_string(line);

        // attribute extraction
        if (line.find("<PP_HEADER") != std::string::npos) {
            // element
            size_t pos = line.find("element=");
            if (pos != std::string::npos) {
                upf.element = line.substr(pos + 9, 2);
            }
            // z_valence
            if (line.find("z_valence=") != std::string::npos){
                sscanf(line.c_str(), "%*[^z]z_valence=\"%lf\"", &upf.z_valence);
            }
            // ultrasoft
            if (line.find("is_ultrasoft=\"T\"") != std::string::npos || line.find("is_ultrasoft=\"true\"") != std::string::npos){
                upf.ultrasoft = true;
            }
            // PAW
            if (line.find("is_paw=\"T\"") != std::string::npos || line.find("is_paw=\"true\"") != std::string::npos){
                upf.paw = true;
            }
            size_t meshpos = line.find("mesh_size=\"");
            size_t pos2 = line.find("number_of_proj=\"");
            if (meshpos != std::string::npos){
                size_t start = meshpos + 11;
                size_t end = line.find("\"", start);
                if (end != std::string::npos) {
                    upf.mesh = std::stoi(line.substr(start, end - start));
                }
            }
            if (pos2 != std::string::npos){
                size_t start = pos2 + 16;
                size_t end = line.find("\"", start);
                if (end != std::string::npos) {
                    upf.nproj = std::stoi(line.substr(start, end - start));
                }
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
            if (line.find("<PP_R>") != std::string::npos){
                upf.r = read_data_for_tag(file, "PP_R");
            }
            if (line.find("<PP_RAB>") != std::string::npos){
                upf.rab = read_data_for_tag(file, "PP_RAB");
            }
        }

        if (line.find("<PP_LOCAL") != std::string::npos){
            upf.vloc = read_data_for_tag(file, "PP_LOCAL");
        }

        // projectors
        if (nonlocal && line.find("<PP_BETA.") != std::string::npos) {
            size_t pos3 = line.find("<PP_BETA.");
            size_t pos4 = line.find("angular_momentum=");

            if (pos3 != std::string::npos && pos4 != std::string::npos) {
                // Extract beta index
                size_t idx_start = pos3 + 9;
                size_t idx_end = line.find_first_not_of("0123456789", idx_start);
                int betaindex = std::stoi(line.substr(idx_start, idx_end - idx_start)) - 1;

                // Extract angular momentum
                size_t l_start = pos4 + 18;
                size_t l_end = line.find("\"", l_start);
                int angularmomentum = std::stoi(line.substr(l_start, l_end - l_start));

                std::vector<double> rawbeta = read_data_for_tag(file, "PP_BETA." + std::to_string(betaindex + 1));
                std::vector<double> beta(upf.mesh);
                for (int i = 0; i < upf.mesh; ++i) {
                    beta[i] = (i < rawbeta.size() && rawbeta[i] != 0.0) ? rawbeta[i] / upf.r[i] : 0.0;
                }
                upf.beta.push_back(beta);
                upf.beta_l.push_back(angularmomentum);
            }
        }

        // D_IJ matrix
        if (nonlocal && line.find("<PP_DIJ>") != std::string::npos) {
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

        // Q matrix (scalar integrals)
        if (nonlocal && line.find("<PP_Q>") != std::string::npos) {
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

        // Parse Q_IJ(r) augmentation functions for ultrasoft
        if (aug && line.find("<PP_QIJL.") != std::string::npos) {
            // Extract indices: <PP_QIJL.i.j.l ...>
            // Format: <PP_QIJL.1.1.0 first_index="1" second_index="1" ...>
            size_t start = line.find("<PP_QIJL.") + 9;  // Start after "<PP_QIJL."
            size_t dotpos1 = line.find(".", start);
            size_t dotpos2 = line.find(".", dotpos1 + 1);
            size_t endpos = line.find_first_of(" >", dotpos2 + 1);  // Find space or > after last index

            // Check for valid positions
            if (dotpos1 != std::string::npos && dotpos2 != std::string::npos &&
                endpos != std::string::npos) {

                int i = std::stoi(line.substr(start, dotpos1 - start)) - 1;  // convert to 0-indexed
                int j = std::stoi(line.substr(dotpos1 + 1, dotpos2 - dotpos1 - 1)) - 1;
                int l = std::stoi(line.substr(dotpos2 + 1, endpos - dotpos2 - 1));

                // Initialize qfunc if needed
                if (upf.qfunc.empty()) {
                    upf.qfunc.assign(upf.nproj, std::vector<std::map<int, std::vector<double>>>(upf.nproj));
                }

                // Read the radial function - construct tag name from indices
                std::string tag = "PP_QIJL." + std::to_string(i + 1) + "." + std::to_string(j + 1) + "." + std::to_string(l);
                std::vector<double> qfunc_data = read_data_for_tag(file, tag);

                // Store for both (i,j) and (j,i) with angular momentum l
                upf.qfunc[i][j][l] = qfunc_data;
                upf.qfunc[j][i][l] = qfunc_data;  // symmetric
                upf.nqf++;
            }
        }

        // Alternative format: <PP_QIJ> with composite index (less common)
        // Note: Must check for "<PP_QIJ." (with dot) to avoid matching "<PP_QIJL."
        if (aug && line.find("<PP_QIJ.") != std::string::npos && line.find("composite_index=") != std::string::npos
            && line.find("<PP_QIJL.") == std::string::npos) {
            // Extract composite index and first/second index
            size_t cpos = line.find("composite_index=");
            size_t fpos = line.find("first_index=");
            size_t spos = line.find("second_index=");

            // Check for valid positions
            if (cpos != std::string::npos && fpos != std::string::npos && spos != std::string::npos) {
                int cidx = std::stoi(line.substr(cpos + 17, 2));
                int i = std::stoi(line.substr(fpos + 13, 1)) - 1;
                int j = std::stoi(line.substr(spos + 14, 1)) - 1;
                int l = 0;  // Default angular momentum

                // Try to extract angular momentum if present
                if (line.find("angular_momentum=") != std::string::npos) {
                    size_t lpos = line.find("angular_momentum=");
                    if (lpos != std::string::npos) {
                        l = std::stoi(line.substr(lpos + 18, 1));
                    }
                }

                // Initialize qfunc if needed
                if (upf.qfunc.empty()) {
                    upf.qfunc.assign(upf.nproj, std::vector<std::map<int, std::vector<double>>>(upf.nproj));
                }

                // Read the radial function
                std::string tag = "PP_QIJ." + std::to_string(cidx);
                std::vector<double> qfunc_data = read_data_for_tag(file, tag);
                upf.qfunc[i][j][l] = qfunc_data;
                upf.qfunc[j][i][l] = qfunc_data;  // symmetric
                upf.nqf++;
            }
        }
    }

    // Post-processing: if ultrasoft but no qfunc, warn
    if (upf.ultrasoft && upf.qfunc.empty()) {
        std::cerr << "WARNING: Ultrasoft pseudopotential but no Q_IJ(r) functions found!\n";
        std::cerr << "         This may be an old UPF format. Augmentation will be disabled.\n";
    }

    return upf;
}

/**
 * Spherical Bessel function j_l(x)
 */
double spherical_bessel(int l, double x) {
    if (x < 1e-8) {
        return (l == 0) ? 1.0 : 0.0;
    }

    if (l == 0) {
        return sin(x) / x;
    } else if (l == 1) {
        return (sin(x) / (x * x)) - (cos(x) / x);
    } else if (l == 2) {
        return ((3.0 / (x * x) - 1.0) * sin(x) / x) - (3.0 * cos(x) / (x * x));
    } else if (l == 3) {
        double x2 = x * x;
        return ((15.0 / (x2 * x) - 6.0 / x) * sin(x)) - ((15.0 / x2 - 1.0) * cos(x));
    }

    // For higher l, use recurrence relation
    double jlm2 = sin(x) / x;  // j_0
    double jlm1 = (sin(x) / (x * x)) - (cos(x) / x);  // j_1
    double jl = 0.0;

    for (int i = 2; i <= l; i++) {
        jl = ((2.0 * i - 1.0) / x) * jlm1 - jlm2;
        jlm2 = jlm1;
        jlm1 = jl;
    }

    return jl;
}

/**
 * Compute pseudopotential matrix element in RYDBERG units
 * Supports multiple projectors for norm-conserving and ultrasoft PSPs
 * V_nl(G,G') = Σ_{i,j} β_i(q) D_{ij} β_j(q)*  where q = |G - G'|
 */
double Vnl_GGp(UPF& upf, double G, double GPrime) {
    double q = std::abs(G - GPrime);  // q = |G - G'| in bohr⁻¹

    // Compute form factors for all projectors: β_i(q) = ∫ β_i(r) j_l(qr) r² dr
    std::vector<double> beta_q(upf.nproj, 0.0);

    for (int iproj = 0; iproj < upf.nproj; iproj++) {
        int l = upf.beta_l[iproj];

        for (int ir = 0; ir < upf.mesh; ir++) {
            if (upf.r[ir] < 1e-12) continue;

            double x = q * upf.r[ir];
            double jl = spherical_bessel(l, x);

            beta_q[iproj] += upf.beta[iproj][ir] * jl * upf.r[ir] * upf.r[ir] * upf.rab[ir];
        }

        beta_q[iproj] *= 4.0 * PI;
    }

    // Sum over all projector pairs: Σ_{i,j} β_i(q) D_{ij} β_j(q)
    double vnl = 0.0;
    for (int i = 0; i < upf.nproj; i++) {
        for (int j = 0; j < upf.nproj; j++) {
            vnl += beta_q[i] * upf.D[i][j] * beta_q[j];
        }
    }

    // Convert from Hartree to Rydberg
    return 2.0 * vnl;
}

/**
 * Compute augmentation charge Q_IJ(q) for ultrasoft pseudopotentials
 * Q_IJ(q) = Σ_l ∫ Q_IJ^l(r) j_l(qr) r² dr
 *
 * For ultrasoft PSPs, the charge density includes an augmentation term:
 * ρ_aug(r) = Σ_{i,j} ρ_ij Q_IJ(r)  where ρ_ij = Σ_n f_n <ψ_n|β_i><β_j|ψ_n>
 *
 * Note: Multiple angular momentum channels may contribute to Q_IJ
 */
double compute_qij_augmentation(UPF& upf, int i, int j, double q) {
    if (!upf.ultrasoft || upf.qfunc.empty()) {
        return 0.0;  // No augmentation for norm-conserving PSPs
    }

    if (i >= upf.nproj || j >= upf.nproj) {
        return 0.0;
    }

    if (upf.qfunc[i][j].empty()) {
        return 0.0;  // No augmentation function for this pair
    }

    double qij_q = 0.0;

    // Sum over all angular momentum channels
    for (const auto& l_func_pair : upf.qfunc[i][j]) {
        int l = l_func_pair.first;
        const std::vector<double>& qfunc_l = l_func_pair.second;

        double qij_q_l = 0.0;
        for (int ir = 0; ir < upf.mesh && ir < qfunc_l.size(); ir++) {
            if (upf.r[ir] < 1e-12) continue;

            double x = q * upf.r[ir];
            double jl = spherical_bessel(l, x);

            qij_q_l += qfunc_l[ir] * jl * upf.r[ir] * upf.r[ir] * upf.rab[ir];
        }

        qij_q_l *= 4.0 * PI;
        qij_q += qij_q_l;
    }

    return qij_q;
}

/**
 * Compute density matrix elements from wavefunctions
 * ρ_ij = Σ_n f_n <ψ_n|β_i><β_j|ψ_n>
 *
 * This function computes the projections of wavefunctions onto the beta projectors.
 * These are needed for ultrasoft augmentation charge computation.
 *
 * @param upf Pseudopotential structure
 * @param wfc Wavefunctions in real space (nstates x mesh)
 * @param occupations Occupation numbers f_n
 * @param nstates Number of states
 * @return Density matrix ρ_ij (nproj x nproj)
 */
std::vector<std::vector<double>> compute_density_matrix(
    UPF& upf,
    const std::vector<std::vector<double>>& wfc,
    const std::vector<double>& occupations,
    int nstates
) {
    std::vector<std::vector<double>> rho_ij(upf.nproj, std::vector<double>(upf.nproj, 0.0));

    // For each state
    for (int n = 0; n < nstates; n++) {
        // Compute projections <β_i|ψ_n> = ∫ β_i(r) ψ_n(r) r² dr
        std::vector<double> beta_psi(upf.nproj, 0.0);

        for (int i = 0; i < upf.nproj; i++) {
            for (int ir = 0; ir < upf.mesh && ir < wfc[n].size(); ir++) {
                beta_psi[i] += upf.beta[i][ir] * wfc[n][ir] * upf.r[ir] * upf.r[ir] * upf.rab[ir];
            }
        }

        // Accumulate ρ_ij += f_n <ψ_n|β_i><β_j|ψ_n>
        for (int i = 0; i < upf.nproj; i++) {
            for (int j = 0; j < upf.nproj; j++) {
                rho_ij[i][j] += occupations[n] * beta_psi[i] * beta_psi[j];
            }
        }
    }

    return rho_ij;
}

/**
 * Compute total augmentation charge in G-space for ultrasoft PSPs
 * ρ_aug(G) = Σ_{i,j} ρ_ij Q_IJ(|G|)
 *
 * @param upf Pseudopotential structure
 * @param rho_ij Density matrix elements <ψ|β_i><β_j|ψ> (nproj x nproj)
 * @param G_magnitude |G| value in bohr⁻¹
 * @return Augmentation charge ρ_aug(G)
 */
double compute_augmentation_charge(UPF& upf, const std::vector<std::vector<double>>& rho_ij, double G) {
    if (!upf.ultrasoft || upf.qfunc.empty()) {
        return 0.0;
    }

    double rho_aug = 0.0;

    for (int i = 0; i < upf.nproj; i++) {
        for (int j = 0; j < upf.nproj; j++) {
            double qij_g = compute_qij_augmentation(upf, i, j, G);
            rho_aug += rho_ij[i][j] * qij_g;
        }
    }

    return rho_aug;
}

/**
 * Pre-compute the full pseudopotential matrix (DENSITY INDEPENDENT!)
 * This should be called ONCE before the SCF loop
 *
 * @param upf Pseudopotential structure
 * @param gvectors Vector of G-vectors
 * @param npw Number of plane waves
 * @return Pre-computed Vnl matrix (npw x npw)
 */
double** precompute_pseudopotential_matrix(UPF& upf, const std::vector<GVector>& gvectors, int npw) {
    std::cout << "  Pre-computing pseudopotential matrix (" << npw << "x" << npw << ")...\n";

    // DIAGNOSTIC: Print pseudopotential information
    std::cout << "\n  PSEUDOPOTENTIAL INFO:\n";
    std::cout << "  Element: " << upf.element << "\n";
    std::cout << "  Type: " << (upf.ultrasoft ? "Ultrasoft" : (upf.paw ? "PAW" : "Norm-conserving")) << "\n";
    std::cout << "  Number of projectors: " << upf.nproj << "\n";
    std::cout << "  Mesh size: " << upf.mesh << "\n";

    if (upf.nproj > 0) {
        std::cout << "  D matrix:\n";
        for (int i = 0; i < upf.nproj; i++) {
            std::cout << "    ";
            for (int j = 0; j < upf.nproj; j++) {
                std::cout << std::setw(12) << std::setprecision(6) << upf.D[i][j] << " ";
            }
            std::cout << "\n";
        }
    }

    if (upf.ultrasoft && !upf.qfunc.empty()) {
        std::cout << "  Augmentation functions: " << upf.nqf << " Q_IJ(r) functions loaded\n";
    }
    std::cout << "\n";

    auto t_start = std::chrono::high_resolution_clock::now();

    // Allocate matrix
    double** Vnl = new double*[npw];
    for (int i = 0; i < npw; i++) {
        Vnl[i] = new double[npw];
    }

    // OPTIMIZATION: Pre-compute beta form factors for all unique q values
    // Build cache of beta_q values: β_i(q) = ∫ β_i(r) j_l(qr) r² dr
    std::cout << "  Building beta form factor cache...\n";
    auto t_cache_start = std::chrono::high_resolution_clock::now();

    // Store cache in UPF structure for later reuse in augmentation charge
    upf.beta_q_cache.clear();
    int cache_hits = 0, cache_misses = 0;

    // First pass: identify all unique q values and compute beta_q for each
    // Need both q = |G-G'| for V_nl AND q = |G| for augmentation projections
    for (int i = 0; i < npw; i++) {
        double Gi = std::sqrt(gvectors[i].g2);

        // Add |Gi| itself to cache (needed for augmentation projections)
        double Gi_rounded = std::round(Gi * 1e10) / 1e10;
        if (upf.beta_q_cache.find(Gi_rounded) == upf.beta_q_cache.end()) {
            std::vector<double> beta_q(upf.nproj, 0.0);
            for (int iproj = 0; iproj < upf.nproj; iproj++) {
                int l = upf.beta_l[iproj];
                for (int ir = 0; ir < upf.mesh; ir++) {
                    if (upf.r[ir] < 1e-12) continue;
                    double x = Gi * upf.r[ir];
                    double jl = spherical_bessel(l, x);
                    beta_q[iproj] += upf.beta[iproj][ir] * jl * upf.r[ir] * upf.r[ir] * upf.rab[ir];
                }
            }
            upf.beta_q_cache[Gi_rounded] = beta_q;
            cache_misses++;
        }

        for (int j = i; j < npw; j++) {
            double Gj = std::sqrt(gvectors[j].g2);
            double q = std::abs(Gi - Gj);

            // Round q to avoid floating point precision issues
            double q_rounded = std::round(q * 1e10) / 1e10;

            if (upf.beta_q_cache.find(q_rounded) == upf.beta_q_cache.end()) {
                // Cache miss - compute beta_q for this q value
                std::vector<double> beta_q(upf.nproj, 0.0);

                for (int iproj = 0; iproj < upf.nproj; iproj++) {
                    int l = upf.beta_l[iproj];

                    for (int ir = 0; ir < upf.mesh; ir++) {
                        if (upf.r[ir] < 1e-12) continue;

                        double x = q * upf.r[ir];
                        double jl = spherical_bessel(l, x);

                        beta_q[iproj] += upf.beta[iproj][ir] * jl * upf.r[ir] * upf.r[ir] * upf.rab[ir];
                    }

                    // NOTE: UPF format already includes 4π normalization
                    // beta_q[iproj] *= 4.0 * PI;
                }

                upf.beta_q_cache[q_rounded] = beta_q;
                cache_misses++;
            }
        }
    }

    auto t_cache_end = std::chrono::high_resolution_clock::now();
    double time_cache = std::chrono::duration<double>(t_cache_end - t_cache_start).count();

    std::cout << "  Beta cache built: " << upf.beta_q_cache.size() << " unique q values in "
              << std::fixed << std::setprecision(2) << time_cache << " s\n";
    std::cout << "  Theoretical speedup: ~" << ((npw * (npw + 1)) / 2.0) / upf.beta_q_cache.size() << "x\n";

    // DIAGNOSTIC: Print sample beta_q values to check normalization
    if (!upf.beta_q_cache.empty()) {
        std::cout << "\n  DIAGNOSTIC - Beta form factor validation:\n";

        // Sample at q=0 (or smallest q)
        auto it_min = upf.beta_q_cache.begin();
        std::cout << "  At q = " << std::fixed << std::setprecision(6) << it_min->first << " (smallest q):\n";
        for (int ip = 0; ip < std::min(upf.nproj, 6); ip++) {
            std::cout << "    beta_" << ip << " (l=" << upf.beta_l[ip] << ") = "
                      << std::scientific << std::setprecision(3) << it_min->second[ip] << "\n";
        }

        // Sample at medium q (middle of cache)
        auto it_mid = upf.beta_q_cache.begin();
        std::advance(it_mid, upf.beta_q_cache.size() / 2);
        std::cout << "  At q = " << std::fixed << std::setprecision(6) << it_mid->first << " (medium q):\n";
        for (int ip = 0; ip < std::min(upf.nproj, 6); ip++) {
            std::cout << "    beta_" << ip << " (l=" << upf.beta_l[ip] << ") = "
                      << std::scientific << std::setprecision(3) << it_mid->second[ip] << "\n";
        }

        // Sample at largest q
        auto it_max = std::prev(upf.beta_q_cache.end());
        std::cout << "  At q = " << std::fixed << std::setprecision(6) << it_max->first << " (largest q):\n";
        for (int ip = 0; ip < std::min(upf.nproj, 6); ip++) {
            std::cout << "    beta_" << ip << " (l=" << upf.beta_l[ip] << ") = "
                      << std::scientific << std::setprecision(3) << it_max->second[ip] << "\n";
        }

        // Validation checks
        std::cout << "\n  Validation checks:\n";

        // Check 1: At q=0, j_l(0) = 1 for l=0, 0 for l>0
        if (it_min->first < 1e-6) {
            std::cout << "  [PASS] q~0 detected: ";
            bool correct = true;
            for (int ip = 0; ip < upf.nproj; ip++) {
                if (upf.beta_l[ip] > 0 && std::abs(it_min->second[ip]) > 1e-6) {
                    correct = false;
                    break;
                }
            }
            if (correct) {
                std::cout << "beta_i(q=0) = 0 for l>0 [OK]\n";
            } else {
                std::cout << "WARNING: beta_i(q=0) should be 0 for l>0!\n";
            }
        }

        // Check 2: Beta values should decay for large q
        // Only check projectors that are non-zero at small q (l=0 projectors)
        bool decays = true;
        int num_checked = 0;
        for (int ip = 0; ip < upf.nproj; ip++) {
            // Skip projectors that are zero at q≈0 (these are l>0)
            if (std::abs(it_min->second[ip]) < 1e-6) continue;

            num_checked++;
            // For non-zero projectors, check if they grow unreasonably at large q
            if (std::abs(it_max->second[ip]) > std::abs(it_min->second[ip]) * 3.0) {
                decays = false;
                break;
            }
        }
        if (num_checked > 0 && decays) {
            std::cout << "  [PASS] Beta form factors decay/stable at large q [OK]\n";
        } else if (num_checked > 0 && !decays) {
            std::cout << "  [WARN] Beta form factors may grow unexpectedly at large q\n";
        } else {
            std::cout << "  [INFO] No l=0 projectors to check decay (all projectors are l>0)\n";
        }

        // Check 3: No NaN or Inf values
        bool has_invalid = false;
        for (const auto& pair : upf.beta_q_cache) {
            for (double val : pair.second) {
                if (std::isnan(val) || std::isinf(val)) {
                    has_invalid = true;
                    break;
                }
            }
            if (has_invalid) break;
        }
        if (!has_invalid) {
            std::cout << "  [PASS] No NaN/Inf values detected [OK]\n";
        } else {
            std::cout << "  [FAIL] ERROR: NaN or Inf values found in cache!\n";
        }

        std::cout << "\n";
    }

    // Track min/max/avg for diagnostics
    double vnl_min = 1e99, vnl_max = -1e99, vnl_sum = 0.0;
    int count = 0;

    // Second pass: Compute all matrix elements using cached beta values
    std::cout << "  Computing matrix elements using cache...\n";
    for (int i = 0; i < npw; i++) {
        double Gi = std::sqrt(gvectors[i].g2);

        for (int j = i; j < npw; j++) {  // Start from j=i to only compute upper triangle
            double Gj = std::sqrt(gvectors[j].g2);
            double q = std::abs(Gi - Gj);
            double q_rounded = std::round(q * 1e10) / 1e10;

            // Retrieve cached beta_q values
            const std::vector<double>& beta_q = upf.beta_q_cache[q_rounded];

            // Compute Vnl = Σ_{i,j} β_i(q) D_{ij} β_j(q)
            // Note: D matrix in UPF format is already in Rydberg units
            double vnl = 0.0;
            for (int ip = 0; ip < upf.nproj; ip++) {
                for (int jp = 0; jp < upf.nproj; jp++) {
                    vnl += beta_q[ip] * upf.D[ip][jp] * beta_q[jp];
                }
            }

            // D matrix is already in Rydberg for this UPF file
            double val = vnl;

            // Store in both positions due to symmetry
            Vnl[i][j] = val;
            Vnl[j][i] = val;

            // Track statistics
            vnl_min = std::min(vnl_min, val);
            vnl_max = std::max(vnl_max, val);
            vnl_sum += val;
            count++;
        }

        // Progress indicator every 10%
        if ((i + 1) % (npw / 10 + 1) == 0) {
            std::cout << "    Progress: " << (100 * (i + 1) / npw) << "%\n";
        }
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double time_pseudo = std::chrono::duration<double>(t_end - t_start).count();
    double time_matrix = time_pseudo - time_cache;

    std::cout << "  Pseudopotential matrix computed in " << std::fixed << std::setprecision(2)
              << time_pseudo << " s\n";
    std::cout << "    - Cache construction: " << time_cache << " s\n";
    std::cout << "    - Matrix computation: " << time_matrix << " s\n";
    std::cout << "  (This is done ONCE - will be reused in all SCF iterations)\n\n";

    // Print diagnostic statistics
    std::cout << "  PSEUDOPOTENTIAL MATRIX STATISTICS:\n";
    std::cout << "  Min value: " << std::scientific << std::setprecision(3) << vnl_min << " Ry\n";
    std::cout << "  Max value: " << std::scientific << std::setprecision(3) << vnl_max << " Ry\n";
    std::cout << "  Avg value: " << std::scientific << std::setprecision(3) << (vnl_sum / count) << " Ry\n";
    std::cout << "  Matrix elements computed: " << count << " (exploiting symmetry)\n";
    std::cout << "  Total matrix size: " << npw << "x" << npw << " (" << (npw*npw) << " elements)\n";
    std::cout << "  Unique q values cached: " << upf.beta_q_cache.size() << "\n";
    std::cout << "  Combined speedup (symmetry + caching): ~"
              << std::fixed << std::setprecision(1)
              << (2.0 * (npw * (npw + 1)) / 2.0) / upf.beta_q_cache.size() << "x\n\n";

    // Validation checks for Vnl matrix
    std::cout << "  VALIDATION CHECKS:\n";

    // Check 1: Values in reasonable range
    double max_abs = std::max(std::abs(vnl_min), std::abs(vnl_max));
    if (max_abs < 100.0) {
        std::cout << "  [PASS] Matrix values in reasonable range (O(1-10) Ry) [OK]\n";
    } else if (max_abs < 1000.0) {
        std::cout << "  [WARN] Matrix values somewhat large (O(100) Ry)\n";
    } else {
        std::cout << "  [FAIL] ERROR: Matrix values too large (O(1000+) Ry)!\n";
    }

    // Check 2: Symmetry verification (spot check)
    bool symmetric = true;
    int sym_checks = std::min(100, npw);
    for (int check = 0; check < sym_checks; check++) {
        int i = rand() % npw;
        int j = rand() % npw;
        if (std::abs(Vnl[i][j] - Vnl[j][i]) > 1e-10) {
            symmetric = false;
            break;
        }
    }
    if (symmetric) {
        std::cout << "  [PASS] Matrix symmetry verified (spot check) [OK]\n";
    } else {
        std::cout << "  [FAIL] ERROR: Matrix is not symmetric!\n";
    }

    // Check 3: No NaN or Inf in final matrix
    bool matrix_valid = true;
    for (int i = 0; i < npw && matrix_valid; i++) {
        for (int j = 0; j < npw; j++) {
            if (std::isnan(Vnl[i][j]) || std::isinf(Vnl[i][j])) {
                matrix_valid = false;
                break;
            }
        }
    }
    if (matrix_valid) {
        std::cout << "  [PASS] No NaN/Inf values in matrix [OK]\n";
    } else {
        std::cout << "  [FAIL] ERROR: NaN or Inf values in Vnl matrix!\n";
    }

    // Check 4: Diagonal elements check
    double diag_sum = 0.0;
    for (int i = 0; i < npw; i++) {
        diag_sum += Vnl[i][i];
    }
    double diag_avg = diag_sum / npw;
    std::cout << "  [INFO] Diagonal average: " << std::scientific << std::setprecision(3)
              << diag_avg << " Ry\n";

    std::cout << "\n";
    
    return Vnl;
}

/**
 * Compute local pseudopotential in G-space for each plane wave
 * V_loc(G) = (4π/Ω) × Σ_atoms [∫ V_loc(r) j_0(|G|r) r² dr] × exp(iG·τ_atom)
 *
 * @param upf Pseudopotential structure
 * @param gvectors Vector of G-vectors
 * @param npw Number of plane waves
 * @param ion_pos_x X-coordinates of atoms (in Bohr)
 * @param ion_pos_y Y-coordinates of atoms (in Bohr)
 * @param ion_pos_z Z-coordinates of atoms (in Bohr)
 * @param natoms Number of atoms
 * @param cell_volume Unit cell volume (Bohr³)
 * @return V_loc(G) for each G-vector (vector of npw complex numbers)
 */
std::vector<std::complex<double>> compute_vloc_g(UPF& upf, const std::vector<GVector>& gvectors, int npw,
                                                  const std::vector<double>& ion_pos_x,
                                                  const std::vector<double>& ion_pos_y,
                                                  const std::vector<double>& ion_pos_z,
                                                  int natoms, double cell_volume) {
    std::cout << "  Computing local pseudopotential in G-space...\n";
    auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<std::complex<double>> vloc_g(npw);

    // For each G-vector
    for (int ig = 0; ig < npw; ig++) {
        double G_mag = std::sqrt(gvectors[ig].g2);

        // Special treatment for G=0: skip it (divergent Coulomb term)
        // The long-range -Z_v/r term is handled by the Ewald summation instead
        if (G_mag < 1e-8) {
            vloc_g[ig] = std::complex<double>(0.0, 0.0);
            continue;
        }

        // Compute radial integral: ∫ V_loc(r) j_0(Gr) r² dr
        double vloc_radial = 0.0;
        for (int ir = 0; ir < upf.mesh; ir++) {
            if (upf.r[ir] < 1e-12) continue;

            double x = G_mag * upf.r[ir];
            double j0 = sin(x) / x;  // j_0(x) = sin(x)/x for x > 0

            vloc_radial += upf.vloc[ir] * j0 * upf.r[ir] * upf.r[ir] * upf.rab[ir];
        }

        vloc_radial *= 4.0 * PI / cell_volume;

        // Sum over all atoms with structure factor exp(iG·τ)
        std::complex<double> vloc_g_total(0.0, 0.0);
        for (int iat = 0; iat < natoms; iat++) {
            double phase = gvectors[ig].gx * ion_pos_x[iat] +
                          gvectors[ig].gy * ion_pos_y[iat] +
                          gvectors[ig].gz * ion_pos_z[iat];

            std::complex<double> structure_factor(cos(phase), sin(phase));
            vloc_g_total += structure_factor * vloc_radial;
        }

        vloc_g[ig] = vloc_g_total;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double time_vloc = std::chrono::duration<double>(t_end - t_start).count();

    // Statistics
    double vloc_min = 1e99, vloc_max = -1e99;
    for (int ig = 0; ig < npw; ig++) {
        double val = vloc_g[ig].real();  // For diagonal elements, it's real
        vloc_min = std::min(vloc_min, val);
        vloc_max = std::max(vloc_max, val);
    }

    std::cout << "  V_loc computed in " << std::fixed << std::setprecision(2) << time_vloc << " s\n";
    std::cout << "  V_loc range: [" << std::scientific << std::setprecision(3)
              << vloc_min << ", " << vloc_max << "] Ry\n\n";

    return vloc_g;
}

/**
 * Compute and add ultrasoft augmentation charge to density (plane-wave version)
 * This computes ρ_aug(G) and adds it to rho_g via FFT
 *
 * @param upf Pseudopotential structure (must have beta_q_cache filled)
 * @param gvectors G-vectors
 * @param npw Number of plane waves
 * @param psi Wavefunctions psi[band][G]
 * @param nocc Number of occupied bands
 * @param ion_pos_x, ion_pos_y, ion_pos_z Ion positions (in Bohr)
 * @param natoms Number of atoms
 * @param rho_g Density in G-space (FFT grid) - will be modified
 * @param nr1, nr2, nr3 FFT grid dimensions
 * @param cell_volume Cell volume
 */
void add_augmentation_charge_planewave(
    UPF& upf,
    const std::vector<GVector>& gvectors,
    int npw,
    const std::vector<std::vector<std::complex<double>>>& psi,
    int nocc,
    const std::vector<double>& ion_pos_x,
    const std::vector<double>& ion_pos_y,
    const std::vector<double>& ion_pos_z,
    int natoms,
    fftw_complex* rho_g,
    int nr1, int nr2, int nr3,
    double cell_volume
) {
    if (!upf.ultrasoft || upf.qfunc.empty()) {
        return;  // Not ultrasoft
    }

    std::cout << "  Adding ultrasoft augmentation charge...\n";
    std::cout << "    Number of projectors: " << upf.nproj << "\n";
    std::cout << "    Number of atoms: " << natoms << "\n";
    std::cout << "    Number of occupied bands: " << nocc << "\n";

    int nproj = upf.nproj;

    // Step 1: Compute projections ⟨β_i|ψ_n⟩ in reciprocal space
    // ⟨β_i|ψ_n⟩ = Σ_G β_i(G) conj(ψ_n(G)) * conj(S(G))
    std::vector<std::vector<std::vector<std::complex<double>>>> beta_proj(
        natoms, std::vector<std::vector<std::complex<double>>>(
            nproj, std::vector<std::complex<double>>(nocc, 0.0)
        )
    );

    for (int iat = 0; iat < natoms; iat++) {
        double tau_x = ion_pos_x[iat];
        double tau_y = ion_pos_y[iat];
        double tau_z = ion_pos_z[iat];

        for (int iproj = 0; iproj < nproj; iproj++) {
            for (int ibnd = 0; ibnd < nocc; ibnd++) {
                std::complex<double> proj(0.0, 0.0);

                for (int ig = 0; ig < npw; ig++) {
                    double G_mag = std::sqrt(gvectors[ig].g2);
                    double G_rounded = std::round(G_mag * 1e10) / 1e10;

                    // Get β_i(G) from cache
                    auto it = upf.beta_q_cache.find(G_rounded);
                    if (it == upf.beta_q_cache.end()) continue;
                    const std::vector<double>& beta_G = it->second;

                    // Structure factor S(G) = exp(iG·τ)
                    double phase = gvectors[ig].gx * tau_x +
                                   gvectors[ig].gy * tau_y +
                                   gvectors[ig].gz * tau_z;
                    std::complex<double> S_G(cos(phase), sin(phase));

                    // ⟨β_i|ψ_n⟩ += β_i(G) * conj(ψ_n(G) * S(G))
                    proj += beta_G[iproj] * std::conj(psi[ibnd][ig] * S_G);
                }

                beta_proj[iat][iproj][ibnd] = proj;
            }
        }
    }

    // DIAGNOSTIC: Check projection magnitudes
    double max_proj = 0.0;
    for (int iat = 0; iat < natoms; iat++) {
        for (int iproj = 0; iproj < nproj; iproj++) {
            for (int ibnd = 0; ibnd < nocc; ibnd++) {
                double mag = std::abs(beta_proj[iat][iproj][ibnd]);
                max_proj = std::max(max_proj, mag);
            }
        }
    }
    std::cout << "    Max projection magnitude: " << std::scientific << max_proj << "\n";

    // Step 2: Build density matrix ρ_ij = Σ_n f_n ⟨ψ_n|β_i⟩⟨β_j|ψ_n⟩
    std::vector<std::vector<std::vector<double>>> rho_ij(
        natoms, std::vector<std::vector<double>>(
            nproj, std::vector<double>(nproj, 0.0)
        )
    );

    for (int iat = 0; iat < natoms; iat++) {
        for (int i = 0; i < nproj; i++) {
            for (int j = 0; j < nproj; j++) {
                double rho = 0.0;
                for (int ibnd = 0; ibnd < nocc; ibnd++) {
                    double f_n = 2.0;  // Occupation (spin degeneracy)
                    rho += f_n * std::real(std::conj(beta_proj[iat][i][ibnd]) * beta_proj[iat][j][ibnd]);
                }
                rho_ij[iat][i][j] = rho;
            }
        }
    }

    // DIAGNOSTIC: Check density matrix magnitudes
    double max_rho_ij = 0.0;
    for (int iat = 0; iat < natoms; iat++) {
        for (int i = 0; i < nproj; i++) {
            for (int j = 0; j < nproj; j++) {
                max_rho_ij = std::max(max_rho_ij, std::abs(rho_ij[iat][i][j]));
            }
        }
    }
    std::cout << "    Max density matrix element: " << std::scientific << max_rho_ij << "\n";

    // Step 3: Add ρ_aug(G) to density in G-space
    double max_aug = 0.0;
    for (int ig = 0; ig < npw; ig++) {
        double G_mag = std::sqrt(gvectors[ig].g2);
        std::complex<double> rho_aug_g(0.0, 0.0);

        for (int iat = 0; iat < natoms; iat++) {
            // Compute Q_IJ(|G|) contribution using existing function
            double qij_total = compute_augmentation_charge(upf, rho_ij[iat], G_mag);

            // Structure factor S(G) = exp(iG·τ)
            double phase = gvectors[ig].gx * ion_pos_x[iat] +
                           gvectors[ig].gy * ion_pos_y[iat] +
                           gvectors[ig].gz * ion_pos_z[iat];
            std::complex<double> S_G(cos(phase), sin(phase));

            rho_aug_g += qij_total * S_G;
        }

        max_aug = std::max(max_aug, std::abs(rho_aug_g));

        // Map to FFT grid and add
        int h = gvectors[ig].h;
        int k = gvectors[ig].k;
        int l = gvectors[ig].l;

        int ih = (h >= 0) ? h : h + nr1;
        int ik = (k >= 0) ? k : k + nr2;
        int il = (l >= 0) ? l : l + nr3;

        if (ih >= 0 && ih < nr1 && ik >= 0 && ik < nr2 && il >= 0 && il < nr3) {
            int idx = ih * (nr2 * nr3) + ik * nr3 + il;
            // Scale by 1/Ω to match density normalization
            rho_g[idx][0] += rho_aug_g.real() / cell_volume;
            rho_g[idx][1] += rho_aug_g.imag() / cell_volume;
        }
    }

    std::cout << "    Max augmentation charge |ρ_aug(G)|: " << std::scientific << max_aug << "\n";
    std::cout << "    (scaled by 1/Ω = 1/" << cell_volume << " when added)\n";
    std::cout << "  Augmentation charge added\n";
}

/**
 * Compute S-overlap matrix for ultrasoft pseudopotentials
 * S_ij = δ_ij + Σ_atoms Σ_{nm} ⟨G_i|β_n⟩ Q_nm(|G_i - G_j|) ⟨β_m|G_j⟩
 *      = δ_ij + Σ_atoms Σ_{nm} β_n(|G_i|) Q_nm(|G_i - G_j|) β_m(|G_j|) exp(i(G_i - G_j)·τ)
 *
 * For ultrasoft PP, must solve generalized eigenvalue problem: H|ψ⟩ = E S|ψ⟩
 *
 * @param upf Pseudopotential (must have beta_q_cache and qfunc)
 * @param gvectors G-vectors
 * @param npw Number of plane waves
 * @param ion_pos_x Ion x-positions (Bohr)
 * @param ion_pos_y Ion y-positions (Bohr)
 * @param ion_pos_z Ion z-positions (Bohr)
 * @param natoms Number of atoms
 * @return S matrix (npw x npw)
 */
double** compute_overlap_matrix(UPF& upf, const std::vector<GVector>& gvectors, int npw,
                                const std::vector<double>& ion_pos_x,
                                const std::vector<double>& ion_pos_y,
                                const std::vector<double>& ion_pos_z,
                                int natoms) {
    std::cout << "  Computing S-overlap matrix for ultrasoft PP (" << npw << "x" << npw << ")...\n";
    auto t_start = std::chrono::high_resolution_clock::now();

    // Allocate S matrix initialized to identity
    double** S = new double*[npw];
    for (int i = 0; i < npw; i++) {
        S[i] = new double[npw];
        for (int j = 0; j < npw; j++) {
            S[i][j] = (i == j) ? 1.0 : 0.0;  // Start with identity matrix
        }
    }

    int nproj = upf.nproj;

    // OPTIMIZATION: Build cache of Q_nm(q) values for all unique q = |G_i - G_j|
    // First pass: identify all unique q values we'll need
    std::cout << "    Identifying unique |G_i - G_j| values...\n";
    std::set<double> unique_q;
    for (int i = 0; i < npw; i++) {
        for (int j = i; j < npw; j++) {
            double dG_x = gvectors[i].gx - gvectors[j].gx;
            double dG_y = gvectors[i].gy - gvectors[j].gy;
            double dG_z = gvectors[i].gz - gvectors[j].gz;
            double q = std::sqrt(dG_x*dG_x + dG_y*dG_y + dG_z*dG_z);
            double q_rounded = std::round(q * 1e10) / 1e10;
            unique_q.insert(q_rounded);
        }
    }

    std::cout << "    Building Q_nm cache for " << unique_q.size() << " unique q values...\n";
    auto t_cache_start = std::chrono::high_resolution_clock::now();

    std::map<double, std::vector<std::vector<double>>> Q_cache;
    double max_Q = 0.0;
    for (double q : unique_q) {
        std::vector<std::vector<double>> Q_matrix(nproj, std::vector<double>(nproj, 0.0));

        for (int n = 0; n < nproj; n++) {
            for (int m = 0; m < nproj; m++) {
                Q_matrix[n][m] = compute_qij_augmentation(upf, n, m, q);
                max_Q = std::max(max_Q, std::abs(Q_matrix[n][m]));
            }
        }
        Q_cache[q] = Q_matrix;
    }

    auto t_cache_end = std::chrono::high_resolution_clock::now();
    double time_cache = std::chrono::duration<double>(t_cache_end - t_cache_start).count();
    std::cout << "    Q_nm cache built in " << std::fixed << std::setprecision(2) << time_cache << " s\n";
    std::cout << "    Max |Q_nm(q)| across all q: " << std::scientific << std::setprecision(3) << max_Q << "\n";

    // For each pair of plane waves (i, j)
    std::cout << "    Computing S-matrix elements...\n";
    for (int i = 0; i < npw; i++) {
        double Gi = std::sqrt(gvectors[i].g2);
        double Gi_rounded = std::round(Gi * 1e10) / 1e10;

        // Check if beta_Gi exists in cache
        auto it_Gi = upf.beta_q_cache.find(Gi_rounded);
        if (it_Gi == upf.beta_q_cache.end()) {
            std::cerr << "ERROR: beta_Gi not found in cache for Gi=" << Gi << "\n";
            continue;
        }
        const std::vector<double>& beta_Gi = it_Gi->second;

        for (int j = i; j < npw; j++) {  // Exploit symmetry
            double Gj = std::sqrt(gvectors[j].g2);
            double Gj_rounded = std::round(Gj * 1e10) / 1e10;

            auto it_Gj = upf.beta_q_cache.find(Gj_rounded);
            if (it_Gj == upf.beta_q_cache.end()) {
                std::cerr << "ERROR: beta_Gj not found in cache for Gj=" << Gj << "\n";
                continue;
            }
            const std::vector<double>& beta_Gj = it_Gj->second;

            // Compute |G_i - G_j| for Q_nm lookup
            double dG_x = gvectors[i].gx - gvectors[j].gx;
            double dG_y = gvectors[i].gy - gvectors[j].gy;
            double dG_z = gvectors[i].gz - gvectors[j].gz;
            double q = std::sqrt(dG_x*dG_x + dG_y*dG_y + dG_z*dG_z);
            double q_rounded = std::round(q * 1e10) / 1e10;

            // Get cached Q_nm matrix for this q
            const std::vector<std::vector<double>>& Q_matrix = Q_cache.at(q_rounded);

            double s_correction = 0.0;

            // Sum over atoms
            for (int iat = 0; iat < natoms; iat++) {
                // Structure factor: exp(i(G_i - G_j)·τ)
                double phase = (gvectors[i].gx - gvectors[j].gx) * ion_pos_x[iat] +
                              (gvectors[i].gy - gvectors[j].gy) * ion_pos_y[iat] +
                              (gvectors[i].gz - gvectors[j].gz) * ion_pos_z[iat];

                double cos_phase = cos(phase);

                // Sum over projectors: Σ_{nm} β_n(|G_i|) Q_nm(|G_i - G_j|) β_m(|G_j|)
                for (int n = 0; n < nproj; n++) {
                    for (int m = 0; m < nproj; m++) {
                        s_correction += beta_Gi[n] * Q_matrix[n][m] * beta_Gj[m] * cos_phase;
                    }
                }
            }

            S[i][j] += s_correction;
            S[j][i] = S[i][j];  // Symmetric
        }

        // Progress
        if ((i + 1) % (npw / 10 + 1) == 0) {
            std::cout << "    Progress: " << (100 * (i + 1) / npw) << "%\n";
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double time_s = std::chrono::duration<double>(t_end - t_start).count();

    // Diagnostics
    double s_min = 1e99, s_max = -1e99;
    double diag_min = 1e99, diag_max = -1e99;
    for (int i = 0; i < npw; i++) {
        diag_min = std::min(diag_min, S[i][i]);
        diag_max = std::max(diag_max, S[i][i]);
        for (int j = 0; j < npw; j++) {
            s_min = std::min(s_min, S[i][j]);
            s_max = std::max(s_max, S[i][j]);
        }
    }

    std::cout << "  S-overlap matrix computed in " << std::fixed << std::setprecision(2) << time_s << " s\n";
    std::cout << "  Diagonal range: [" << std::scientific << std::setprecision(3)
              << diag_min << ", " << diag_max << "]\n";
    std::cout << "  Full matrix range: [" << s_min << ", " << s_max << "]\n";
    std::cout << "  (Identity would be diagonal=1.0, off-diagonal=0.0)\n";

    // Check for positive definiteness issues
    int neg_diag = 0;
    int zero_diag = 0;
    for (int i = 0; i < npw; i++) {
        if (S[i][i] <= 0.0) neg_diag++;
        if (std::abs(S[i][i]) < 1e-10) zero_diag++;
    }
    if (neg_diag > 0) {
        std::cout << "  [ERROR] " << neg_diag << " diagonal elements are non-positive!\n";
    }
    if (zero_diag > 0) {
        std::cout << "  [WARN] " << zero_diag << " diagonal elements are near zero\n";
    }
    std::cout << "\n";

    return S;
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