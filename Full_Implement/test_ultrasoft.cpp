#include <iostream>
#include <iomanip>
#include "pseudopotentials.cpp"

/**
 * Test program for ultrasoft pseudopotential parsing
 */

int main() {
    std::cout << "========================================\n";
    std::cout << "  Ultrasoft Pseudopotential Parser Test\n";
    std::cout << "========================================\n\n";

    std::string upf_file = "Fe.UPF";

    try {
        std::cout << "Reading UPF file: " << upf_file << "\n\n";
        UPF upf = read_upf(upf_file);

        // Print header information
        std::cout << "HEADER INFORMATION:\n";
        std::cout << "-------------------\n";
        std::cout << "Element:           " << upf.element << "\n";
        std::cout << "Z valence:         " << upf.z_valence << "\n";
        std::cout << "Ultrasoft:         " << (upf.ultrasoft ? "YES" : "NO") << "\n";
        std::cout << "PAW:               " << (upf.paw ? "YES" : "NO") << "\n";
        std::cout << "Mesh size:         " << upf.mesh << "\n";
        std::cout << "Number of proj:    " << upf.nproj << "\n";
        std::cout << "L max:             " << upf.lmax << "\n\n";

        // Print radial grid information
        if (!upf.r.empty()) {
            std::cout << "RADIAL GRID:\n";
            std::cout << "------------\n";
            std::cout << "Grid points:       " << upf.r.size() << "\n";
            std::cout << "r[0] =             " << std::scientific << std::setprecision(6) << upf.r[0] << "\n";
            std::cout << "r[mid] =           " << upf.r[upf.r.size()/2] << "\n";
            std::cout << "r[max] =           " << upf.r.back() << "\n\n";
        }

        // Print local potential information
        if (!upf.vloc.empty()) {
            std::cout << "LOCAL POTENTIAL:\n";
            std::cout << "----------------\n";
            std::cout << "V_loc points:      " << upf.vloc.size() << "\n";
            std::cout << "V_loc[0] =         " << std::fixed << std::setprecision(6) << upf.vloc[0] << "\n";
            std::cout << "V_loc[mid] =       " << upf.vloc[upf.vloc.size()/2] << "\n\n";
        }

        // Print beta projector information
        std::cout << "BETA PROJECTORS:\n";
        std::cout << "----------------\n";
        std::cout << "Number of beta:    " << upf.beta.size() << "\n";
        for (int i = 0; i < upf.beta.size(); i++) {
            std::cout << "  Beta[" << i << "]: l=" << upf.beta_l[i]
                      << ", size=" << upf.beta[i].size() << "\n";
        }
        std::cout << "\n";

        // Print D matrix
        if (!upf.D.empty()) {
            std::cout << "D MATRIX:\n";
            std::cout << "---------\n";
            std::cout << "Dimension: " << upf.D.size() << "x" << upf.D[0].size() << "\n";
            for (int i = 0; i < upf.D.size(); i++) {
                std::cout << "  ";
                for (int j = 0; j < upf.D[i].size(); j++) {
                    std::cout << std::setw(12) << std::setprecision(6) << upf.D[i][j] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }

        // Print Q matrix (scalar integrals)
        if (!upf.Q.empty()) {
            std::cout << "Q MATRIX (scalar integrals):\n";
            std::cout << "----------------------------\n";
            std::cout << "Dimension: " << upf.Q.size() << "x" << upf.Q[0].size() << "\n";
            for (int i = 0; i < upf.Q.size(); i++) {
                std::cout << "  ";
                for (int j = 0; j < upf.Q[i].size(); j++) {
                    std::cout << std::setw(12) << std::setprecision(6) << upf.Q[i][j] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }

        // Print augmentation function information (ULTRASOFT SPECIFIC)
        if (upf.ultrasoft && !upf.qfunc.empty()) {
            std::cout << "AUGMENTATION FUNCTIONS (Q_IJ):\n";
            std::cout << "-------------------------------\n";
            std::cout << "Number of Q_IJ entries: " << upf.nqf << "\n";
            std::cout << "Q_IJ functions loaded:\n";

            for (int i = 0; i < upf.nproj; i++) {
                for (int j = i; j < upf.nproj; j++) {
                    if (!upf.qfunc[i][j].empty()) {
                        std::cout << "  Q[" << i << "][" << j << "]: ";
                        for (const auto& l_func : upf.qfunc[i][j]) {
                            std::cout << "l=" << l_func.first
                                      << " (size=" << l_func.second.size() << ") ";
                        }
                        std::cout << "\n";
                    }
                }
            }
            std::cout << "\n";

            // Test augmentation charge computation at q=1.0 bohr^-1
            std::cout << "TESTING AUGMENTATION CHARGE COMPUTATION:\n";
            std::cout << "----------------------------------------\n";
            double test_q = 1.0;
            std::cout << "Computing Q_IJ(q) at q = " << test_q << " bohr^-1\n\n";

            std::cout << "Q_IJ(q) matrix:\n";
            for (int i = 0; i < upf.nproj; i++) {
                std::cout << "  ";
                for (int j = 0; j < upf.nproj; j++) {
                    double qij = compute_qij_augmentation(upf, i, j, test_q);
                    std::cout << std::setw(12) << std::setprecision(6) << qij << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        } else if (upf.ultrasoft) {
            std::cout << "WARNING: Ultrasoft PSP but no Q_IJ functions found!\n";
            std::cout << "This may be an old UPF format.\n\n";
        }

        // Test spherical Bessel functions
        std::cout << "TESTING SPHERICAL BESSEL FUNCTIONS:\n";
        std::cout << "------------------------------------\n";
        double x = 2.0;
        std::cout << "At x = " << x << ":\n";
        std::cout << "  j_0(x) = " << std::setprecision(8) << spherical_bessel(0, x) << "\n";
        std::cout << "  j_1(x) = " << spherical_bessel(1, x) << "\n";
        std::cout << "  j_2(x) = " << spherical_bessel(2, x) << "\n";
        std::cout << "  j_3(x) = " << spherical_bessel(3, x) << "\n\n";

        std::cout << "========================================\n";
        std::cout << "TEST COMPLETED SUCCESSFULLY!\n";
        std::cout << "========================================\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
