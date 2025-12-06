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
                upf.mesh = std::stoi(line.substr(meshpos + 11, 4));
            }
            if (pos2 != std::string::npos){
                upf.nproj = std::stoi(line.substr(pos2 + 16, 1));
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
            int n, l = 0;
            size_t pos3 = line.find("<PP_BETA.");
            size_t pos4 = line.find("angular_momentum=");
            int betaindex = std::stoi(line.substr(pos3 + 9, 1)) - 1;
            int angularmomentum = std::stoi(line.substr(pos4 + 18, 1));
            std::vector<double> rawbeta = read_data_for_tag(file, "PP_BETA." + std::to_string(betaindex + 1));
            std::vector<double> beta(upf.mesh);
            for (int i = 0; i < upf.mesh; ++i) {
                beta[i] = (i < rawbeta.size() && rawbeta[i] != 0.0) ? rawbeta[i] / upf.r[i] : 0.0;
            }
            upf.beta.push_back(beta);
            upf.beta_l.push_back(angularmomentum);
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

        // Q matrix
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
    }

    return upf;
}

/**
 * Compute pseudopotential matrix element in RYDBERG units
 * FIXED: Converted from Hartree to Rydberg (multiply by 2)
 */
double Vnl_GGp(UPF& upf, double G, double GPrime) {
    double q = abs(G - GPrime);           // q = |G - G'| in bohr⁻¹
    double form_factor = 0.0;
    
    for (int i = 0; i < upf.r.size(); ++i) {
        if (upf.r[i] < 1e-12) continue;
        double x = q * upf.r[i];
        double j0 = (x < 1e-8) ? 1.0 : sin(x)/x;        // spherical Bessel j₀(x)
        form_factor += upf.beta[0][i] * j0 * upf.r[i]*upf.r[i] * upf.rab[i];  // ∫ β(r) j₀(qr) r² dr
    }
    
    form_factor *= 4.0 * PI;
    
    // CRITICAL FIX: Convert from Hartree to Rydberg by multiplying by 2
    return 2.0 * upf.D[0][0] * form_factor;   // Now in Rydberg units
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
    
    // DIAGNOSTIC: Print D[0][0] value
    std::cout << "\n  DIAGNOSTIC INFO:\n";
    std::cout << "  D[0][0] = " << upf.D[0][0] << " (should be ~0.1-1.0 in Hartree)\n";
    std::cout << "  D[0][0] * 2 = " << (2.0 * upf.D[0][0]) << " (Rydberg conversion)\n\n";
    
    auto t_start = std::chrono::high_resolution_clock::now();
    
    // Allocate matrix
    double** Vnl = new double*[npw];
    for (int i = 0; i < npw; i++) {
        Vnl[i] = new double[npw];
    }
    
    // Track min/max/avg for diagnostics
    double vnl_min = 1e99, vnl_max = -1e99, vnl_sum = 0.0;
    int count = 0;
    
    // Compute all matrix elements
    for (int i = 0; i < npw; i++) {
        double Gi = std::sqrt(gvectors[i].g2);
        
        for (int j = 0; j < npw; j++) {
            double Gj = std::sqrt(gvectors[j].g2);
            Vnl[i][j] = Vnl_GGp(upf, Gi, Gj);
            
            // Track statistics
            vnl_min = std::min(vnl_min, Vnl[i][j]);
            vnl_max = std::max(vnl_max, Vnl[i][j]);
            vnl_sum += Vnl[i][j];
            count++;
        }
        
        // Progress indicator every 10%
        if ((i + 1) % (npw / 10 + 1) == 0) {
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