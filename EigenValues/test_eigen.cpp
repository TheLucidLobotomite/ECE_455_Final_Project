#include "cpu_eigen.cpp"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <iostream>
#include <iomanip>

// Forward declarations for GPU functions
extern "C" {
    EigenResult* compute_eigenvalues_gpu(double** matrix, int n);
    // Note: free_eigen_result is already declared in cpu_eigen.cpp
}

// ============= TEST MATRICES =============

// Create sparse Hamiltonian (tridiagonal + random sparse elements)
double** create_sparse_hamiltonian(int n, double sparsity, unsigned int seed) {
    srand(seed);
    double** H = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        H[i] = (double*)calloc(n, sizeof(double));
    }
    
    // Diagonal: on-site energies
    for (int i = 0; i < n; i++) {
        H[i][i] = 1.0 + ((double)rand() / RAND_MAX) * 4.0;
    }
    
    // Off-diagonal: nearest neighbor hopping (symmetric)
    for (int i = 0; i < n-1; i++) {
        double val = -0.1 - ((double)rand() / RAND_MAX) * 0.9;
        H[i][i+1] = val;
        H[i+1][i] = val;
    }
    
    // Add sparse random interactions (symmetric)
    int n_random = (int)(n * n * sparsity);
    for (int r = 0; r < n_random; r++) {
        int i = rand() % n;
        int j = rand() % n;
        if (i != j) {
            double val = ((double)rand() / RAND_MAX - 0.5) * 1.0;
            H[i][j] = val;
            H[j][i] = val;  // Ensure symmetry
        }
    }
    
    return H;
}

// ============= UTILITIES =============

void free_matrix(double** mat, int n) {
    for (int i = 0; i < n; i++) free(mat[i]);
    free(mat);
}

/**
 * Compare CPU and GPU eigenvalue results
 */
void compare_results(const std::vector<double>& cpu_eigenvalues,
                    const std::vector<double>& gpu_eigenvalues,
                    int n) {
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    double avg_abs_diff = 0.0;
    
    for (int i = 0; i < n; i++) {
        double abs_diff = std::abs(cpu_eigenvalues[i] - gpu_eigenvalues[i]);
        double rel_diff = abs_diff / (std::abs(cpu_eigenvalues[i]) + 1e-12);
        
        avg_abs_diff += abs_diff;
        
        if (abs_diff > max_abs_diff) {
            max_abs_diff = abs_diff;
        }
        if (rel_diff > max_rel_diff) {
            max_rel_diff = rel_diff;
        }
    }
    
    avg_abs_diff /= n;
    
    std::cout << "\n  Comparison Results:\n";
    std::cout << "  " << std::string(70, '-') << "\n";
    std::cout << "    Max absolute difference: " << std::scientific << std::setprecision(3) 
              << max_abs_diff << "\n";
    std::cout << "    Max relative difference: " << std::fixed << std::setprecision(6) 
              << (max_rel_diff * 100.0) << " %\n";
    std::cout << "    Avg absolute difference: " << std::scientific << std::setprecision(3) 
              << avg_abs_diff << "\n";
    
    // Tolerance check
    const double abs_tol = 1e-6;
    const double rel_tol = 1e-6;
    
    bool passed = (max_abs_diff < abs_tol) && (max_rel_diff < rel_tol);
    
    if (passed) {
        std::cout << "    Results MATCH within tolerance\n";
    } else {
        std::cout << "    Results DIFFER beyond tolerance\n";
        std::cout << "      (tolerance: " << std::scientific << abs_tol 
                  << " absolute, " << std::fixed << (rel_tol * 100.0) << "% relative)\n";
    }
}

// ============= DETAILED COMPARISON TEST =============

void run_comparison_test(int n, int num_runs, double sparsity) {
    std::cout << "\n========================================\n";
    std::cout << "Detailed Comparison: " << n << "x" << n << " matrix\n";
    std::cout << "========================================\n";
    std::cout << "Number of runs: " << num_runs << "\n";
    std::cout << "Sparsity: " << (sparsity * 100.0) << "%\n\n";
    
    std::vector<double> cpu_times_ms;
    std::vector<double> gpu_times_ms;
    
    std::vector<double> cpu_eigenvalues;
    std::vector<double> gpu_eigenvalues;
    
    // ==================== CPU BENCHMARK ====================
    std::cout << "[CPU] Running computation...\n";
    
    for (int run = 0; run < num_runs; run++) {
        unsigned int seed = 42 + run;
        double** A = create_sparse_hamiltonian(n, sparsity, seed);
        
        auto cpu_start = std::chrono::high_resolution_clock::now();
        EigenResult* cpu_result = compute_eigenvalues(A, n);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        
        double cpu_time_ms = std::chrono::duration<double>(cpu_end - cpu_start).count() * 1000.0;
        cpu_times_ms.push_back(cpu_time_ms);
        
        // Save eigenvalues from last run for comparison
        if (run == num_runs - 1) {
            for (int i = 0; i < n; i++) {
                cpu_eigenvalues.push_back(cpu_result->values[i]);
            }
        }
        
        free_eigen_result(cpu_result);
        free_matrix(A, n);
    }
    
    double cpu_avg = 0.0;
    double cpu_min = cpu_times_ms[0];
    double cpu_max = cpu_times_ms[0];
    
    for (double t : cpu_times_ms) {
        cpu_avg += t;
        if (t < cpu_min) cpu_min = t;
        if (t > cpu_max) cpu_max = t;
    }
    cpu_avg /= num_runs;
    
    std::cout << "  Average time: " << std::fixed << std::setprecision(2) << cpu_avg << " ms\n";
    std::cout << "  Min time:     " << cpu_min << " ms\n";
    std::cout << "  Max time:     " << cpu_max << " ms\n";
    
    // ==================== GPU BENCHMARK ====================
    std::cout << "\n[GPU] Running computation...\n";
    
    for (int run = 0; run < num_runs; run++) {
        unsigned int seed = 42 + run;
        double** A = create_sparse_hamiltonian(n, sparsity, seed);
        
        auto gpu_start = std::chrono::high_resolution_clock::now();
        EigenResult* gpu_result = compute_eigenvalues_gpu(A, n);
        auto gpu_end = std::chrono::high_resolution_clock::now();
        
        double gpu_time_ms = std::chrono::duration<double>(gpu_end - gpu_start).count() * 1000.0;
        gpu_times_ms.push_back(gpu_time_ms);
        
        // Save eigenvalues from last run for comparison
        if (run == num_runs - 1) {
            for (int i = 0; i < n; i++) {
                gpu_eigenvalues.push_back(gpu_result->values[i]);
            }
        }
        
        free_eigen_result(gpu_result);
        free_matrix(A, n);
    }
    
    double gpu_avg = 0.0;
    double gpu_min = gpu_times_ms[0];
    double gpu_max = gpu_times_ms[0];
    
    for (double t : gpu_times_ms) {
        gpu_avg += t;
        if (t < gpu_min) gpu_min = t;
        if (t > gpu_max) gpu_max = t;
    }
    gpu_avg /= num_runs;
    
    std::cout << "  Average time: " << std::fixed << std::setprecision(2) << gpu_avg << " ms\n";
    std::cout << "  Min time:     " << gpu_min << " ms\n";
    std::cout << "  Max time:     " << gpu_max << " ms\n";
    
    double speedup = cpu_avg / gpu_avg;
    std::cout << "\n  GPU Speedup:  " << std::setprecision(2) << speedup << "x\n";
    
    // ==================== COMPARISON ====================
    compare_results(cpu_eigenvalues, gpu_eigenvalues, n);
}

// ============= CONVERGENCE TEST =============

struct BenchmarkResult {
    int size;
    int num_runs;
    double cpu_avg_ms;
    double gpu_avg_ms;
    double speedup;
};

void run_convergence_test() {
    std::cout << "\n========================================\n";
    std::cout << "Convergence & Performance Scaling\n";
    std::cout << "========================================\n\n";
    
    const double sparsity = 0.01;
    
    // Test configurations
    struct TestConfig {
        int size;
        int num_runs;
    };
    
    std::vector<TestConfig> tests = {
        {50, 20},       // Small - 20 runs
        {100, 20},      // Medium-small - 20 runs
        {150, 10},      // Medium - 10 runs
        {175, 5},       // Medium-large - 5 runs
        {250, 3},       // Large - 3 runs
        {1000, 2},      // Very large - 2 runs
        {5000, 1},      // Huge - 1 run
        {10000, 1}      // Massive - 1 run
    };
    
    std::vector<BenchmarkResult> results;
    
    std::cout << std::setw(12) << "Matrix Size"
              << std::setw(10) << "Runs"
              << std::setw(18) << "CPU Avg (ms)"
              << std::setw(18) << "GPU Avg (ms)"
              << std::setw(14) << "Speedup\n";
    std::cout << std::string(72, '-') << "\n";
    
    for (const auto& test : tests) {
        int n = test.size;
        int num_runs = test.num_runs;
        
        std::cout << std::fixed << std::setprecision(0);
        std::cout << std::setw(10) << n << "x" << n
                  << std::setw(10) << num_runs << std::flush;
        
        std::vector<double> cpu_times;
        std::vector<double> gpu_times;
        
        // CPU benchmark
        for (int run = 0; run < num_runs; run++) {
            unsigned int seed = 42 + run;
            double** A = create_sparse_hamiltonian(n, sparsity, seed);
            
            auto t0 = std::chrono::high_resolution_clock::now();
            EigenResult* r = compute_eigenvalues(A, n);
            auto t1 = std::chrono::high_resolution_clock::now();
            
            double ms = std::chrono::duration<double>(t1 - t0).count() * 1000.0;
            cpu_times.push_back(ms);
            
            free_eigen_result(r);
            free_matrix(A, n);
        }
        
        // GPU benchmark
        for (int run = 0; run < num_runs; run++) {
            unsigned int seed = 42 + run;
            double** A = create_sparse_hamiltonian(n, sparsity, seed);
            
            auto t0 = std::chrono::high_resolution_clock::now();
            EigenResult* r = compute_eigenvalues_gpu(A, n);
            auto t1 = std::chrono::high_resolution_clock::now();
            
            double ms = std::chrono::duration<double>(t1 - t0).count() * 1000.0;
            gpu_times.push_back(ms);
            
            free_eigen_result(r);
            free_matrix(A, n);
        }
        
        // Calculate averages
        double cpu_avg = 0.0, gpu_avg = 0.0;
        for (double t : cpu_times) cpu_avg += t;
        for (double t : gpu_times) gpu_avg += t;
        cpu_avg /= num_runs;
        gpu_avg /= num_runs;
        
        double speedup = cpu_avg / gpu_avg;
        
        BenchmarkResult result;
        result.size = n;
        result.num_runs = num_runs;
        result.cpu_avg_ms = cpu_avg;
        result.gpu_avg_ms = gpu_avg;
        result.speedup = speedup;
        results.push_back(result);
        
        std::cout << std::setprecision(2);
        std::cout << std::setw(18) << cpu_avg
                  << std::setw(18) << gpu_avg
                  << std::setw(14) << speedup << "\n";
    }
    
    std::cout << std::string(72, '-') << "\n";
}

// ============= MAIN =============

int main() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Eigensolver Performance Comparison\n";
    std::cout << "CPU (LAPACK) vs GPU (cuSOLVER)\n";
    std::cout << "========================================\n";
    std::cout << "Matrix type: Sparse symmetric Hamiltonian\n";
    std::cout << "Sparsity: 1% off-diagonal elements\n";
    std::cout << "CPU Solver: LAPACK dsyevd\n";
    std::cout << "GPU Solver: cuSOLVER Dsyevd\n";
    std::cout << "========================================\n";
    
    const double sparsity = 0.01;
    
    // Detailed comparison tests
    std::cout << "\n========================================\n";
    std::cout << "DETAILED COMPARISON TESTS\n";
    std::cout << "========================================\n";
    
    run_comparison_test(100, 20, sparsity);
    run_comparison_test(250, 10, sparsity);
    run_comparison_test(1000, 2, sparsity);
    
    // Convergence test
    std::cout << "\n\n========================================\n";
    std::cout << "CONVERGENCE & PERFORMANCE SCALING\n";
    std::cout << "========================================\n";
    
    run_convergence_test();
    
    std::cout << "\n========================================\n";
    std::cout << "All Tests Complete!\n";
    std::cout << "========================================\n\n";
    
    std::cout << "Summary:\n";
    std::cout << "  - GPU speedup is measured for each matrix size\n";
    std::cout << "  - Both solvers should produce matching eigenvalues\n";
    std::cout << "  - Performance scales as O(n^3) for both CPU and GPU\n";
    std::cout << "  - GPU advantage increases with matrix size\n\n";
    
    return 0;
}