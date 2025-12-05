#include <iostream>
#include <iomanip>
#include <fstream>
#include "lobotomites_main.cpp"

/**
 * DFT Test Driver Program
 * Configure and run different DFT calculations here
 */

int main() {
    std::cout << "========================================\n";
    std::cout << "Plane-Wave DFT Code - Test Driver\n";
    std::cout << "  - Kinetic Energy\n";
    std::cout << "  - Exchange-Correlation (LDA)\n";
    std::cout << "  - Hartree Potential\n";
    std::cout << "  - Pseudopotentials (optional)\n";
    std::cout << "========================================\n\n";
    
    // ============================================
    // CONFIGURE YOUR CALCULATION HERE
    // ============================================
    
    DFTContext ctx;
    
    // System: Simple cubic Fe
    double alat = 6.767109;  // Lattice constant in Bohr
    ctx.a1[0] = alat; ctx.a1[1] = 0.0;   ctx.a1[2] = 0.0;
    ctx.a2[0] = 0.0;   ctx.a2[1] = alat; ctx.a2[2] = 0.0;
    ctx.a3[0] = 0.0;   ctx.a3[1] = 0.0;   ctx.a3[2] = alat;
    
    // DFT Parameters
    ctx.ecut_ry = 50.0;           // Energy cutoff (Rydberg)
    ctx.nelec = 8;                // Number of electrons (Fe atom: 8 valence)
    ctx.nbnd = ctx.nelec / 2 + 3; // Number of bands (occupied + some empty)
    
    // SCF Parameters
    ctx.mixing_beta = 0.4;        // Density mixing (0.1-0.5 typical)
    ctx.conv_thr = 1.0e-6;        // Convergence threshold (Ry)
    ctx.max_iter = 50;            // Maximum SCF iterations
    
    // Pseudopotential (optional)
    ctx.use_pseudopot = false;
    ctx.pseudopot = nullptr;
    
    std::string upf_file = "NA.UPF";  // Path to your UPF file
    std::cout << "Checking for pseudopotential: " << upf_file << "\n";
    
    // Check if file exists first
    std::ifstream check_file(upf_file);
    if (check_file.good()) {
        check_file.close();
        try {
            UPF pp = read_upf(upf_file);
            ctx.pseudopot = new UPF(pp);
            ctx.use_pseudopot = true;
            std::cout << "✓ Pseudopotential loaded successfully!\n";
            std::cout << "  Element: " << ctx.pseudopot->element << "\n";
            std::cout << "  Z_valence: " << ctx.pseudopot->z_valence << "\n";
            std::cout << "  Mesh points: " << ctx.pseudopot->mesh << "\n";
            std::cout << "  Projectors: " << ctx.pseudopot->nproj << "\n\n";
        } catch (const std::exception& e) {
            std::cout << "✗ Error loading pseudopotential: " << e.what() << "\n";
            std::cout << "  Continuing with all-electron calculation\n\n";
            ctx.use_pseudopot = false;
        }
    } else {
        std::cout << "⚠ Pseudopotential file not found: " << upf_file << "\n";
        std::cout << "  Continuing with all-electron calculation\n";
        std::cout << "  (Place Fe.UPF in project directory to enable pseudopotentials)\n\n";
        ctx.use_pseudopot = false;
    }
    
    // ============================================
    // PRINT CALCULATION SUMMARY
    // ============================================
    
    std::cout << "Calculation Summary:\n";
    std::cout << "-------------------\n";
    std::cout << "System: Fe atom in cubic cell\n";
    std::cout << "Lattice constant: " << alat << " Bohr\n";
    std::cout << "Energy cutoff: " << ctx.ecut_ry << " Ry\n";
    std::cout << "Electrons: " << ctx.nelec << "\n";
    std::cout << "Bands: " << ctx.nbnd << "\n";
    std::cout << "Mixing beta: " << ctx.mixing_beta << "\n";
    std::cout << "Convergence threshold: " << std::scientific << ctx.conv_thr << " Ry\n";
    std::cout << "Max iterations: " << ctx.max_iter << "\n";
    std::cout << "Pseudopotentials: " << (ctx.use_pseudopot ? "Enabled" : "Disabled") << "\n\n";
    
    // ============================================
    // RUN DFT CALCULATION
    // ============================================
    
    init_reciprocal_lattice(&ctx);
    generate_gvectors(&ctx);
    setup_fft_grid(&ctx);
    init_density(&ctx);
    
    run_scf(&ctx);
    
    // ============================================
    // CLEANUP
    // ============================================
    
    if (ctx.pseudopot != nullptr) {
        delete ctx.pseudopot;
    }
    
    cleanup_dft_context(&ctx);
    
    std::cout << "========================================\n";
    std::cout << "Calculation completed successfully!\n";
    std::cout << "========================================\n";
    
    return 0;
}

/*
 * ============================================
 * HOW TO USE THIS FILE:
 * ============================================
 * 
 * 1. Modify the parameters in the "CONFIGURE YOUR CALCULATION HERE" section
 * 2. Compile and run:
 *    
 *    $env:PATH = "C:\msys64\ucrt64\bin;C:\msys64\ucrt64\lib;" + $env:PATH; 
 *    g++ test_integrated_dft.cpp -o test_dft.exe -std=c++11 -O3 -march=native 
 *    -fopenmp -I"C:\msys64\ucrt64\include" -L"C:\msys64\ucrt64\lib" 
 *    -lfftw3 -lfftw3_threads -lopenblas; 
 *    .\test_dft.exe
 * 
 * ============================================
 * EXAMPLE CONFIGURATIONS:
 * ============================================
 * 
 * Small Test (fast):
 *   ctx.ecut_ry = 10.0;
 *   ctx.nelec = 4;
 *   ctx.max_iter = 20;
 * 
 * Medium Accuracy:
 *   ctx.ecut_ry = 30.0;
 *   ctx.nelec = 8;
 *   ctx.max_iter = 50;
 * 
 * High Accuracy (slow):
 *   ctx.ecut_ry = 50.0;
 *   ctx.nelec = 8;
 *   ctx.max_iter = 100;
 * 
 * Different Materials:
 *   - Change alat (lattice constant)
 *   - Change nelec (number of valence electrons)
 *   - Load appropriate UPF file for pseudopotential
 * 
 */