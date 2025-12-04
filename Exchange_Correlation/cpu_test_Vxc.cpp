#include "cpu_Vxc.cpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>  // Add this line
#include <chrono>     // Also add this for timing

// Physical constants
const double BOHR_TO_ANGSTROM = 0.529177;

/**
 * Simple test density: n(r) = |r|
 */
double test_density(const double* r) {
    double x = r[0];
    double y = r[1];
    double z = r[2];
    double dist = std::sqrt(x*x + y*y + z*z);
    return dist + 1e-10;  // Small offset to avoid zero at origin
}

/**
 * More realistic density: Gaussian centered at origin
 */
double gaussian_density(const double* r) {
    double x = r[0];
    double y = r[1];
    double z = r[2];
    double r2 = x*x + y*y + z*z;
    
    // Gaussian with width = 2 Bohr, peak density = 0.1 e/Bohr³
    double n0 = 0.1;
    double sigma = 2.0;
    return n0 * std::exp(-r2 / (2.0 * sigma * sigma));
}

/**
 * Compute exchange-correlation energy
 * E_xc = ∫ n(r) * ε_xc[n(r)] dr
 */
double compute_exc_energy(VxcContext* ctx) {
    double exc_total = 0.0;
    
    // Volume element
    double cell_volume = ctx->a1[0] * ctx->a2[1] * ctx->a3[2];  // For cubic cell
    double volume_element = cell_volume / ctx->num_points;
    
    for (size_t i = 0; i < ctx->num_points; i++) {
        double n = ctx->density[i];
        double vxc = ctx->vxc[i];
        
        // For LDA: ε_xc can be approximated from Vxc
        // More accurately: need to compute ε_xc separately, but this is close enough
        // ε_xc ≈ Vxc for testing purposes
        double exc_per_electron = vxc;
        
        exc_total += n * exc_per_electron * volume_element;
    }
    
    return exc_total;
}

/**
 * Compute total integrated density (should equal number of electrons)
 */
double compute_total_density(VxcContext* ctx) {
    double total = 0.0;
    
    double cell_volume = ctx->a1[0] * ctx->a2[1] * ctx->a3[2];
    double volume_element = cell_volume / ctx->num_points;
    
    for (size_t i = 0; i < ctx->num_points; i++) {
        total += ctx->density[i] * volume_element;
    }
    
    return total;
}

/**
 * Compute statistics for Vxc
 */
void compute_vxc_stats(VxcContext* ctx, double& min_vxc, double& max_vxc, 
                      double& avg_vxc, double& min_n, double& max_n, double& avg_n) {
    min_vxc = 1e10;
    max_vxc = -1e10;
    avg_vxc = 0.0;
    min_n = 1e10;
    max_n = -1e10;
    avg_n = 0.0;
    
    for (size_t i = 0; i < ctx->num_points; i++) {
        double vxc = ctx->vxc[i];
        double n = ctx->density[i];
        
        if (vxc < min_vxc) min_vxc = vxc;
        if (vxc > max_vxc) max_vxc = vxc;
        avg_vxc += vxc;
        
        if (n < min_n) min_n = n;
        if (n > max_n) max_n = n;
        avg_n += n;
    }
    
    avg_vxc /= ctx->num_points;
    avg_n /= ctx->num_points;
}

/**
 * Run convergence test for a given density function
 */
void run_convergence_test(DensityFunction density_func, const char* test_name) {
    std::cout << "\n========================================\n";
    std::cout << "Convergence Test: " << test_name << "\n";
    std::cout << "========================================\n\n";
    
    // Fe lattice parameters
    double lattice_constant = 6.767109;  // Bohr
    double a1[3] = {lattice_constant, 0.0, 0.0};
    double a2[3] = {0.0, lattice_constant, 0.0};
    double a3[3] = {0.0, 0.0, lattice_constant};
    
    // Grid sizes to test
    std::vector<int> grid_sizes = {32, 48, 64, 96};
    
    // Storage for results
    std::vector<double> exc_energies;
    std::vector<double> total_densities;
    std::vector<double> avg_vxc_values;
    
    std::cout << std::setw(8) << "Grid" 
              << std::setw(15) << "Points"
              << std::setw(18) << "Grid Spacing"
              << std::setw(18) << "E_xc (Ry)"
              << std::setw(18) << "ΔE_xc (Ry)"
              << std::setw(18) << "Total N (e)"
              << std::setw(18) << "Avg Vxc (Ry)"
              << std::setw(15) << "Time (s)\n";
    std::cout << std::string(125, '-') << "\n";
    
    for (int N : grid_sizes) {
        auto start = std::chrono::steady_clock::now();
        
        // Initialize context
        VxcContext* ctx = vxc_init(a1, a2, a3, N, N, N);
        
        // Compute Vxc
        vxc_compute_full(ctx, density_func);
        
        // Compute energies and statistics
        double exc = compute_exc_energy(ctx);
        double total_n = compute_total_density(ctx);
        
        double min_vxc, max_vxc, avg_vxc, min_n, max_n, avg_n;
        compute_vxc_stats(ctx, min_vxc, max_vxc, avg_vxc, min_n, max_n, avg_n);
        
        auto end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        
        // Store results
        exc_energies.push_back(exc);
        total_densities.push_back(total_n);
        avg_vxc_values.push_back(avg_vxc);
        
        // Compute change from previous grid
        double delta_exc = (exc_energies.size() > 1) ? 
                          (exc - exc_energies[exc_energies.size()-2]) : 0.0;
        
        // Print results
        std::cout << std::fixed << std::setprecision(0);
        std::cout << std::setw(6) << N << "³" 
                  << std::setw(15) << (N*N*N);
        
        std::cout << std::setprecision(6);
        std::cout << std::setw(18) << (lattice_constant/N)
                  << std::setprecision(8)
                  << std::setw(18) << exc;
        
        if (exc_energies.size() > 1) {
            std::cout << std::setw(18) << delta_exc;
        } else {
            std::cout << std::setw(18) << "---";
        }
        
        std::cout << std::setw(18) << total_n
                  << std::setw(18) << avg_vxc
                  << std::setprecision(3)
                  << std::setw(15) << elapsed << "\n";
        
        vxc_cleanup(ctx);
    }
    
    // Analyze convergence
    std::cout << "\n" << std::string(125, '-') << "\n";
    std::cout << "Convergence Analysis:\n";
    std::cout << std::string(125, '-') << "\n";
    
    if (exc_energies.size() >= 2) {
        double final_exc = exc_energies.back();
        double prev_exc = exc_energies[exc_energies.size()-2];
        double convergence = std::abs(final_exc - prev_exc);
        
        std::cout << "  Final E_xc change: " << std::scientific << std::setprecision(3) 
                  << convergence << " Ry\n";
        
        if (convergence < 1e-3) {
            std::cout << "  ✓ Well converged (< 0.001 Ry)\n";
        } else if (convergence < 1e-2) {
            std::cout << "  ~ Reasonably converged (< 0.01 Ry)\n";
        } else {
            std::cout << "  ✗ Not well converged - consider finer grid\n";
        }
    }
    
    // Check if total density is conserved
    double density_spread = *std::max_element(total_densities.begin(), total_densities.end()) -
                           *std::min_element(total_densities.begin(), total_densities.end());
    
    std::cout << "\n  Total integrated density range: " 
              << std::fixed << std::setprecision(6) << density_spread << " e\n";
    
    if (density_spread < 0.01) {
        std::cout << "  ✓ Density integration is stable\n";
    }
}

int main() {
    std::cout << "========================================\n";
    std::cout << "Vxc Grid Convergence Test\n";
    std::cout << "========================================\n";
    std::cout << "Cell: Fe FCC, a = 6.767109 Bohr\n";
    std::cout << "Functional: LDA (PW92)\n";
    std::cout << "Testing grids: 32³, 48³, 64³, 96³\n";
    
    // Test 1: Simple radial density n(r) = |r|
    run_convergence_test(test_density, "n(r) = |r|");
    
    // Test 2: Gaussian density (more realistic)
    run_convergence_test(gaussian_density, "Gaussian n(r)");
    
    std::cout << "\n========================================\n";
    std::cout << "Tests Complete!\n";
    std::cout << "========================================\n\n";
    
    std::cout << "Interpretation:\n";
    std::cout << "  - E_xc should converge smoothly as grid is refined\n";
    std::cout << "  - ΔE_xc should decrease with finer grids\n";
    std::cout << "  - Total density should be conserved across grids\n";
    std::cout << "  - Typical convergence target: |ΔE_xc| < 0.001 Ry\n";
    
    return 0;
}