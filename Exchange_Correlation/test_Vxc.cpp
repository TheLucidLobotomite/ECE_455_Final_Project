#include "cpu_Vxc.cpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <omp.h>

// Physical constants
const double BOHR_TO_ANGSTROM = 0.529177;

// Forward declarations for GPU functions
extern "C" {
    struct GpuVxcContext;
    GpuVxcContext* vxc_init(size_t max_size);
    void vxc_compute_pinned(GpuVxcContext* ctx, size_t size);
    void vxc_cleanup(GpuVxcContext* ctx);
}

// Manually declare the GPU context structure
struct GpuVxcContext {
    double *d_n;
    double *d_vxc;
    double *h_n_pinned;
    double *h_vxc_pinned;
    size_t capacity;
};

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
 * Compute Vxc at all grid points (OpenMP parallel version)
 */
void vxc_compute_parallel(VxcContext* ctx) {
    #pragma omp parallel for
    for (size_t i = 0; i < ctx->num_points; i++) {
        ctx->vxc[i] = compute_vxc(ctx->density[i]);
    }
}

/**
 * Complete workflow with parallel computation
 */
void vxc_compute_full_parallel(VxcContext* ctx, DensityFunction density_func) {
    vxc_compute_density(ctx, density_func);
    vxc_compute_parallel(ctx);
}

/**
 * Compute exchange-correlation energy
 */
double compute_exc_energy(VxcContext* ctx) {
    double exc_total = 0.0;
    
    // Volume element
    double cell_volume = ctx->a1[0] * ctx->a2[1] * ctx->a3[2];
    double volume_element = cell_volume / ctx->num_points;
    
    for (size_t i = 0; i < ctx->num_points; i++) {
        double n = ctx->density[i];
        double vxc = ctx->vxc[i];
        double exc_per_electron = vxc;
        exc_total += n * exc_per_electron * volume_element;
    }
    
    return exc_total;
}

/**
 * Compute total integrated density
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
 * Compare two CPU results (single vs parallel)
 */
void compare_cpu_results(const std::vector<double>& cpu1_vxc, 
                        const std::vector<double>& cpu2_vxc,
                        size_t size) {
    double max_abs_diff = 0.0;
    double avg_abs_diff = 0.0;
    
    for (size_t i = 0; i < size; i++) {
        double abs_diff = std::abs(cpu1_vxc[i] - cpu2_vxc[i]);
        avg_abs_diff += abs_diff;
        if (abs_diff > max_abs_diff) {
            max_abs_diff = abs_diff;
        }
    }
    
    avg_abs_diff /= size;
    
    std::cout << "    Single vs Parallel CPU diff: " << std::scientific << std::setprecision(3) 
              << max_abs_diff << " Ry (max), " << avg_abs_diff << " Ry (avg)\n";
}

/**
 * Compare CPU and GPU results
 */
void compare_results(const std::vector<double>& cpu_vxc, 
                    const std::vector<double>& gpu_vxc,
                    size_t size) {
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    double avg_abs_diff = 0.0;
    size_t num_diffs = 0;
    
    for (size_t i = 0; i < size; i++) {
        double abs_diff = std::abs(cpu_vxc[i] - gpu_vxc[i]);
        double rel_diff = abs_diff / (std::abs(cpu_vxc[i]) + 1e-12);
        
        avg_abs_diff += abs_diff;
        num_diffs++;
        
        if (abs_diff > max_abs_diff) {
            max_abs_diff = abs_diff;
        }
        if (rel_diff > max_rel_diff) {
            max_rel_diff = rel_diff;
        }
    }
    
    avg_abs_diff /= num_diffs;
    
    std::cout << "\n  Comparison Results:\n";
    std::cout << "  " << std::string(70, '-') << "\n";
    std::cout << "    Max absolute difference: " << std::scientific << std::setprecision(3) 
              << max_abs_diff << " Ry\n";
    std::cout << "    Max relative difference: " << std::fixed << std::setprecision(6) 
              << (max_rel_diff * 100.0) << " %\n";
    std::cout << "    Avg absolute difference: " << std::scientific << std::setprecision(3) 
              << avg_abs_diff << " Ry\n";
    
    // Tolerance check
    const double abs_tol = 1e-6;  // 1 µRy tolerance
    const double rel_tol = 1e-6;  // 0.0001% tolerance
    
    bool passed = (max_abs_diff < abs_tol) && (max_rel_diff < rel_tol);
    
    if (passed) {
        std::cout << "    ✓ Results MATCH within tolerance\n";
    } else {
        std::cout << "    ✗ Results DIFFER beyond tolerance\n";
        std::cout << "      (tolerance: " << std::scientific << abs_tol 
                  << " Ry absolute, " << std::fixed << (rel_tol * 100.0) << "% relative)\n";
    }
}

/**
 * Run comparison test for single-thread CPU vs parallel CPU vs GPU
 */
void run_comparison_test(DensityFunction density_func, const char* test_name, int N) {
    std::cout << "\n========================================\n";
    std::cout << "Full Comparison: " << test_name << "\n";
    std::cout << "Grid: " << N << "³ = " << (N*N*N) << " points\n";
    std::cout << "========================================\n";
    
    // Fe lattice parameters
    double lattice_constant = 6.767109;  // Bohr
    double a1[3] = {lattice_constant, 0.0, 0.0};
    double a2[3] = {0.0, lattice_constant, 0.0};
    double a3[3] = {0.0, 0.0, lattice_constant};
    
    size_t num_points = N * N * N;
    
    // ==================== SINGLE-THREAD CPU ====================
    std::cout << "\n[CPU Single-Thread] Running computation...\n";
    auto cpu_single_start = std::chrono::steady_clock::now();
    
    VxcContext* cpu_single_ctx = vxc_init(a1, a2, a3, N, N, N);
    vxc_compute_full(cpu_single_ctx, density_func);
    
    auto cpu_single_end = std::chrono::steady_clock::now();
    double cpu_single_time = std::chrono::duration<double>(cpu_single_end - cpu_single_start).count();
    
    std::vector<double> cpu_single_vxc = cpu_single_ctx->vxc;
    std::vector<double> cpu_density = cpu_single_ctx->density;
    
    double cpu_single_exc = compute_exc_energy(cpu_single_ctx);
    double cpu_total_n = compute_total_density(cpu_single_ctx);
    
    std::cout << "  Time: " << std::fixed << std::setprecision(6) << cpu_single_time << " s\n";
    std::cout << "  E_xc: " << std::setprecision(8) << cpu_single_exc << " Ry\n";
    std::cout << "  Total N: " << std::setprecision(6) << cpu_total_n << " e\n";
    
    // ==================== MULTI-THREAD CPU (OpenMP) ====================
    int num_threads = omp_get_max_threads();
    if (num_threads < 8) {
        omp_set_num_threads(8);
        num_threads = 8;
    }
    
    std::cout << "\n[CPU Multi-Thread OpenMP] Running computation (" << num_threads << " threads)...\n";
    auto cpu_parallel_start = std::chrono::steady_clock::now();
    
    VxcContext* cpu_parallel_ctx = vxc_init(a1, a2, a3, N, N, N);
    vxc_compute_full_parallel(cpu_parallel_ctx, density_func);
    
    auto cpu_parallel_end = std::chrono::steady_clock::now();
    double cpu_parallel_time = std::chrono::duration<double>(cpu_parallel_end - cpu_parallel_start).count();
    
    std::vector<double> cpu_parallel_vxc = cpu_parallel_ctx->vxc;
    double cpu_parallel_exc = compute_exc_energy(cpu_parallel_ctx);
    
    std::cout << "  Time: " << std::fixed << std::setprecision(6) << cpu_parallel_time << " s\n";
    std::cout << "  E_xc: " << std::setprecision(8) << cpu_parallel_exc << " Ry\n";
    std::cout << "  Speedup vs single-thread: " << std::setprecision(2) 
              << (cpu_single_time / cpu_parallel_time) << "x\n";
    
    // ==================== GPU COMPUTATION ====================
    std::cout << "\n[GPU] Running computation...\n";
    
    GpuVxcContext* gpu_ctx = vxc_init(num_points);
    
    for (size_t i = 0; i < num_points; i++) {
        gpu_ctx->h_n_pinned[i] = cpu_density[i];
    }
    
    auto gpu_start = std::chrono::steady_clock::now();
    vxc_compute_pinned(gpu_ctx, num_points);
    auto gpu_end = std::chrono::steady_clock::now();
    double gpu_time = std::chrono::duration<double>(gpu_end - gpu_start).count();
    
    std::vector<double> gpu_vxc(num_points);
    for (size_t i = 0; i < num_points; i++) {
        gpu_vxc[i] = gpu_ctx->h_vxc_pinned[i];
    }
    
    double gpu_exc = 0.0;
    double cell_volume = a1[0] * a2[1] * a3[2];
    double volume_element = cell_volume / num_points;
    
    for (size_t i = 0; i < num_points; i++) {
        gpu_exc += cpu_density[i] * gpu_vxc[i] * volume_element;
    }
    
    std::cout << "  Time: " << std::fixed << std::setprecision(6) << gpu_time << " s\n";
    std::cout << "  E_xc: " << std::setprecision(8) << gpu_exc << " Ry\n";
    std::cout << "  Speedup vs single-thread CPU: " << std::setprecision(2) 
              << (cpu_single_time / gpu_time) << "x\n";
    std::cout << "  Speedup vs multi-thread CPU: " << std::setprecision(2) 
              << (cpu_parallel_time / gpu_time) << "x\n";
    
    // ==================== COMPARISONS ====================
    std::cout << "\n  Verification:\n";
    std::cout << "  " << std::string(70, '-') << "\n";
    compare_cpu_results(cpu_single_vxc, cpu_parallel_vxc, num_points);
    
    std::cout << "\n  CPU (parallel) vs GPU Comparison:\n";
    compare_results(cpu_parallel_vxc, gpu_vxc, num_points);
    
    // Cleanup
    vxc_cleanup(cpu_single_ctx);
    vxc_cleanup(cpu_parallel_ctx);
    vxc_cleanup(gpu_ctx);
}

/**
 * Run convergence test comparing all three implementations
 */
void run_convergence_comparison(DensityFunction density_func, const char* test_name) {
    std::cout << "\n========================================\n";
    std::cout << "Convergence Test: " << test_name << "\n";
    std::cout << "========================================\n\n";
    
    // Fe lattice parameters
    double lattice_constant = 6.767109;  // Bohr
    double a1[3] = {lattice_constant, 0.0, 0.0};
    double a2[3] = {0.0, lattice_constant, 0.0};
    double a3[3] = {0.0, 0.0, lattice_constant};
    
    // Grid sizes to test
    std::vector<int> grid_sizes = {16, 32, 64, 128, 256, 512};
    
    int num_threads = omp_get_max_threads();
    if (num_threads < 8) {
        omp_set_num_threads(8);
        num_threads = 8;
    }
    
    std::cout << "OpenMP threads: " << num_threads << "\n\n";
    
    std::cout << std::setw(8) << "Grid" 
              << std::setw(15) << "Points"
              << std::setw(16) << "CPU 1T (s)"
              << std::setw(16) << "CPU MT (s)"
              << std::setw(16) << "GPU (s)"
              << std::setw(14) << "MT Speedup"
              << std::setw(14) << "GPU Speedup"
              << std::setw(18) << "E_xc (Ry)"
              << std::setw(18) << "ΔE_xc (Ry)\n";
    std::cout << std::string(145, '-') << "\n";
    
    for (int N : grid_sizes) {
        size_t num_points = N * N * N;
        
        // Single-thread CPU
        auto cpu_single_start = std::chrono::steady_clock::now();
        VxcContext* cpu_single_ctx = vxc_init(a1, a2, a3, N, N, N);
        vxc_compute_full(cpu_single_ctx, density_func);
        auto cpu_single_end = std::chrono::steady_clock::now();
        double cpu_single_time = std::chrono::duration<double>(cpu_single_end - cpu_single_start).count();
        double cpu_exc = compute_exc_energy(cpu_single_ctx);
        
        // Multi-thread CPU
        auto cpu_parallel_start = std::chrono::steady_clock::now();
        VxcContext* cpu_parallel_ctx = vxc_init(a1, a2, a3, N, N, N);
        vxc_compute_full_parallel(cpu_parallel_ctx, density_func);
        auto cpu_parallel_end = std::chrono::steady_clock::now();
        double cpu_parallel_time = std::chrono::duration<double>(cpu_parallel_end - cpu_parallel_start).count();
        double cpu_parallel_exc = compute_exc_energy(cpu_parallel_ctx);
        
        // GPU computation
        GpuVxcContext* gpu_ctx = vxc_init(num_points);
        for (size_t i = 0; i < num_points; i++) {
            gpu_ctx->h_n_pinned[i] = cpu_single_ctx->density[i];
        }
        
        auto gpu_start = std::chrono::steady_clock::now();
        vxc_compute_pinned(gpu_ctx, num_points);
        auto gpu_end = std::chrono::steady_clock::now();
        double gpu_time = std::chrono::duration<double>(gpu_end - gpu_start).count();
        
        double gpu_exc = 0.0;
        double cell_volume = a1[0] * a2[1] * a3[2];
        double volume_element = cell_volume / num_points;
        for (size_t i = 0; i < num_points; i++) {
            gpu_exc += cpu_single_ctx->density[i] * gpu_ctx->h_vxc_pinned[i] * volume_element;
        }
        
        double mt_speedup = cpu_single_time / cpu_parallel_time;
        double gpu_speedup = cpu_single_time / gpu_time;
        double delta_exc = std::abs(cpu_exc - gpu_exc);
        
        // Print results
        std::cout << std::fixed << std::setprecision(0);
        std::cout << std::setw(6) << N << "³" 
                  << std::setw(15) << num_points;
        
        std::cout << std::setprecision(6);
        std::cout << std::setw(16) << cpu_single_time
                  << std::setw(16) << cpu_parallel_time
                  << std::setw(16) << gpu_time;
        
        std::cout << std::setprecision(2);
        std::cout << std::setw(14) << mt_speedup
                  << std::setw(14) << gpu_speedup;
        
        std::cout << std::setprecision(8);
        std::cout << std::setw(18) << cpu_exc
                  << std::scientific << std::setprecision(3)
                  << std::setw(18) << delta_exc << "\n";
        
        vxc_cleanup(cpu_single_ctx);
        vxc_cleanup(cpu_parallel_ctx);
        vxc_cleanup(gpu_ctx);
    }
    
    std::cout << std::string(145, '-') << "\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "Vxc Performance Comparison\n";
    std::cout << "CPU (Single) vs CPU (OpenMP) vs GPU\n";
    std::cout << "========================================\n";
    std::cout << "Cell: Fe FCC, a = 6.767109 Bohr\n";
    std::cout << "Functional: LDA (PW92)\n\n";
    
    // Detailed comparison tests
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "DETAILED COMPARISON TESTS\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    
    run_comparison_test(test_density, "n(r) = |r|", 64);
    run_comparison_test(gaussian_density, "Gaussian n(r)", 64);
    
    // Convergence comparison
    std::cout << "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "CONVERGENCE & PERFORMANCE SCALING\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    
    run_convergence_comparison(test_density, "n(r) = |r|");
    run_convergence_comparison(gaussian_density, "Gaussian n(r)");
    
    std::cout << "\n========================================\n";
    std::cout << "All Tests Complete!\n";
    std::cout << "========================================\n\n";
    
    std::cout << "Summary:\n";
    std::cout << "  - Single-thread vs Multi-thread shows OpenMP scaling\n";
    std::cout << "  - GPU speedup is measured against single-thread baseline\n";
    std::cout << "  - All implementations should match within 1e-6 Ry\n";
    
    return 0;
}