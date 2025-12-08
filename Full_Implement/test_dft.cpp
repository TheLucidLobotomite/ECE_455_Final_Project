#include <iostream>
#include <iomanip>
#include <fstream>
#include "lobotomites_main_cpu.cpp"
//#include "lobotomites_main_gpu.cpp"

/**
 * DFT Test Driver Program - WITH PSEUDOPOTENTIALS AND EWALD
 * Configure and run different DFT calculations here
 */

int main() {
    std::cout << "========================================\n";
    std::cout << "    Plane-Wave DFT Code - Test Driver\n";
    std::cout << "========================================\n\n";
    
    // ============================================
    // CONFIGURE YOUR CALCULATION HERE
    // ============================================
    
    DFTContext ctx;

    // System: BCC Iron (2 atoms per unit cell)
    double alat = 5.42;  // BCC Fe lattice constant in Bohr (~2.87 Å)
    ctx.a1[0] = alat; ctx.a1[1] = 0.0;   ctx.a1[2] = 0.0;
    ctx.a2[0] = 0.0;   ctx.a2[1] = alat; ctx.a2[2] = 0.0;
    ctx.a3[0] = 0.0;   ctx.a3[1] = 0.0;   ctx.a3[2] = alat;

    // Ion positions and charges (for Ewald summation)
    // BCC structure: atoms at (0,0,0) and (a/2, a/2, a/2)
    ctx.ion_positions = {
        {0.0, 0.0, 0.0},           // First Fe atom
        {alat/2.0, alat/2.0, alat/2.0}  // Second Fe atom (body center)
    };
    ctx.ion_charges = {26.0, 26.0};  // Fe has 26 protons (nuclear charge)

    // DFT Parameters
    ctx.ecut_ry = 71.0;           // Energy cutoff (Rydberg) - reduced for faster testing
    ctx.nelec = 16;               // Number of electrons (2 Fe atoms × 8 valence e⁻)
    ctx.nbnd = ctx.nelec / 2 + 4; // Number of bands (occupied + some empty)
    
    // SCF Parameters
    ctx.mixing_beta = 0.5;        // Density mixing (0.1-0.5 typical)
    ctx.conv_thr = 1.0e-6;        // Convergence threshold (Ry)
    ctx.max_iter = 50;            // Maximum SCF iterations
    
    // Pseudopotential (optional)
    ctx.use_pseudopot = false;
    ctx.pseudopot = nullptr;
    ctx.Vnl_matrix = nullptr;     // Initialize to nullptr
    
    std::string upf_file = "Fe.UPF";  // Path to your UPF file
    std::cout << "Checking for pseudopotential: " << upf_file << "\n";
    
    // Check if file exists first
    std::ifstream check_file(upf_file);
    if (check_file.good()) {
        check_file.close();
        try {
            UPF pp = read_upf(upf_file);
            ctx.pseudopot = new UPF(pp);
            ctx.use_pseudopot = true;
            std::cout << "  Pseudopotential loaded successfully!\n";
            std::cout << "  Element: " << ctx.pseudopot->element << "\n";
            std::cout << "  Z_valence: " << ctx.pseudopot->z_valence << "\n";
            std::cout << "  Mesh points: " << ctx.pseudopot->mesh << "\n";
            std::cout << "  Projectors: " << ctx.pseudopot->nproj << "\n\n";
        } catch (const std::exception& e) {
            std::cout << "  Error loading pseudopotential: " << e.what() << "\n";
            std::cout << "  Continuing with all-electron calculation\n\n";
            ctx.use_pseudopot = false;
        }
    } else {
        std::cout << "  Pseudopotential file not found: " << upf_file << "\n";
        std::cout << "  Continuing with all-electron calculation\n";
        std::cout << "  (Place .UPF in project directory to enable pseudopotentials)\n\n";
        ctx.use_pseudopot = false;
    }
    
    // ============================================
    // PRINT CALCULATION SUMMARY
    // ============================================
    
    std::cout << "Calculation Summary:\n";
    std::cout << "-------------------\n";
    std::cout << "System: BCC Iron (2 atoms)\n";
    std::cout << "Lattice constant: " << alat << " Bohr (" << alat*0.529177 << " Angstrom)\n";
    std::cout << "Energy cutoff: " << ctx.ecut_ry << " Ry\n";
    std::cout << "Electrons: " << ctx.nelec << "\n";
    std::cout << "Bands: " << ctx.nbnd << "\n";
    std::cout << "Ions: " << ctx.ion_positions.size() << " Fe atoms\n";
    std::cout << "Mixing beta: " << ctx.mixing_beta << "\n";
    std::cout << "Convergence threshold: " << std::scientific << ctx.conv_thr << " Ry\n";
    std::cout << "Max iterations: " << ctx.max_iter << "\n";
    std::cout << "Pseudopotentials: " << (ctx.use_pseudopot ? "Enabled (FIXED)" : "Disabled") << "\n\n";
    
    // ============================================
    // SETUP CALCULATION
    // ============================================
    
    init_reciprocal_lattice(&ctx);
    generate_gvectors(&ctx);
    setup_fft_grid(&ctx);
    
    // PRE-COMPUTE PSEUDOPOTENTIAL MATRIX (density-independent!)
    // This happens ONCE before SCF loop
    setup_pseudopotential(&ctx);
    
    // PRE-COMPUTE EWALD ENERGY (density-independent!)
    // This happens ONCE before SCF loop
    setup_ewald_energy(&ctx);
    
    init_density(&ctx);
    
    // ============================================
    // RUN DFT CALCULATION
    // ============================================
    
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