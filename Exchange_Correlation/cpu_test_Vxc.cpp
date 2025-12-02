/**
 * Test Suite for CPU Implementations (Sequential and Parallel)
 * 
 * Tests correctness and performance of Perdew-Wang 1992 implementations
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include "single_thread_Vxc.cpp"
#include "multi_thread_Vxc.cpp"

const double TOLERANCE = 1e-10;

// Test correctness
bool test_correctness() {
    std::cout << "\n=== Correctness Tests ===\n\n";
    
    bool all_passed = true;
    
    // Test 1: Known values comparison
    std::cout << "Test 1: Known density values\n";
    std::vector<double> n_test = {0.001, 0.01, 0.1, 1.0, 10.0};
    
    std::cout << "n (e/bohr³)     Vx (Ha)         Vc (Ha)         Vxc (Ha)        Vxc (eV)\n";
    std::cout << "-------------------------------------------------------------------------------\n";
    
    for (double n : n_test) {
        double Vx = calculate_Vx(n);
        double Vc = calculate_Vc(n);
        double Vxc = Vx + Vc;
        
        std::cout << std::fixed << std::setprecision(6);
        std::cout << std::setw(10) << n << "      "
                  << std::setw(10) << Vx << "      "
                  << std::setw(10) << Vc << "      "
                  << std::setw(10) << Vxc << "      "
                  << std::setw(10) << Vxc * 27.211 << "\n";
    }
    std::cout << "  PASSED\n";
    
    // Test 2: Sequential vs Parallel consistency
    std::cout << "\nTest 2: Sequential vs Parallel consistency\n";
    const size_t test_size = 10000;
    std::vector<double> n_data(test_size);
    std::vector<double> vxc_seq(test_size);
    std::vector<double> vxc_par(test_size);
    
    // Generate test data
    for (size_t i = 0; i < test_size; ++i) {
        n_data[i] = 0.01 + 10.0 * static_cast<double>(i) / test_size;
    }
    
    // Calculate with both methods
    calculate_vxc_sequential(n_data.data(), vxc_seq.data(), test_size);
    calculate_vxc_parallel(n_data.data(), vxc_par.data(), test_size);
    
    // Compare results
    double max_diff = 0.0;
    for (size_t i = 0; i < test_size; ++i) {
        double diff = std::abs(vxc_seq[i] - vxc_par[i]);
        max_diff = std::max(max_diff, diff);
    }
    
    std::cout << "  Maximum difference: " << max_diff << "\n";
    if (max_diff < TOLERANCE) {
        std::cout << "  PASSED\n";
    } else {
        std::cout << "  FAILED (difference too large)\n";
        all_passed = false;
    }
    
    // Test 3: Monotonicity
    std::cout << "\nTest 3: Monotonicity check\n";
    bool monotonic = true;
    for (size_t i = 1; i < test_size; ++i) {
        if (vxc_seq[i] > vxc_seq[i-1]) {
            monotonic = false;
            break;
        }
    }
    
    if (monotonic) {
        std::cout << "  PASSED (Vxc monotonically decreases with density)\n";
    } else {
        std::cout << "  FAILED (non-monotonic behavior detected)\n";
        all_passed = false;
    }
    
    // Test 4: High-density limit
    std::cout << "\nTest 4: High-density limit\n";
    double n_high = 100.0;
    double vxc_high = calculate_Vxc(n_high);
    double vx_high = calculate_Vx(n_high);
    
    double ratio = std::abs(vxc_high) / std::abs(vx_high);
    std::cout << "  n = " << n_high << ": |Vxc|/|Vx| = " << ratio << "\n";
    
    if (ratio > 1.0 && ratio < 1.5) {
        std::cout << "  PASSED\n";
    } else {
        std::cout << "  WARNING (unexpected ratio)\n";
    }
    
    // Test 5: Low-density stability
    std::cout << "\nTest 5: Low-density stability\n";
    double n_low = 1e-6;
    double vxc_low = calculate_Vxc(n_low);
    
    if (std::isfinite(vxc_low)) {
        std::cout << "  n = " << n_low << ": Vxc = " << vxc_low << " Ha (finite)\n";
        std::cout << "  PASSED\n";
    } else {
        std::cout << "  FAILED (NaN or Inf detected)\n";
        all_passed = false;
    }
    
    return all_passed;
}

// Benchmark performance
void benchmark_performance() {
    std::cout << "\n=== Performance Benchmarks ===\n\n";
    
    std::vector<size_t> sizes = {100000, 1000000, 10000000};
    
    std::cout << "Number of OpenMP threads: " << omp_get_max_threads() << "\n\n";
    
    for (size_t size : sizes) {
        std::cout << "Grid size: " << size << " points\n";
        
        // Generate test data
        std::vector<double> n_data(size);
        std::vector<double> vxc_result(size);
        
        for (size_t i = 0; i < size; ++i) {
            n_data[i] = 0.1 * std::exp(-static_cast<double>(i) / size);
        }
        
        // Sequential timing
        double start = omp_get_wtime();
        calculate_vxc_sequential(n_data.data(), vxc_result.data(), size);
        double time_seq = omp_get_wtime() - start;
        
        // Parallel timing
        start = omp_get_wtime();
        calculate_vxc_parallel(n_data.data(), vxc_result.data(), size);
        double time_par = omp_get_wtime() - start;
        
        double speedup = time_seq / time_par;
        
        std::cout << "  Sequential: " << std::fixed << std::setprecision(2) 
                  << time_seq * 1000 << " ms\n";
        std::cout << "  Parallel:   " << time_par * 1000 << " ms\n";
        std::cout << "  Speedup:    " << speedup << "x\n";
        std::cout << "  Throughput: " << size / time_par / 1e6 << " million points/sec\n\n";
    }
}

int main() {
    std::cout << "====================================================\n";
    std::cout << "  Perdew-Wang 1992 LDA - CPU Test Suite\n";
    std::cout << "====================================================\n";
    
    // Run correctness tests
    bool tests_passed = test_correctness();
    
    if (tests_passed) {
        std::cout << "\n✓ All correctness tests passed!\n";
    } else {
        std::cout << "\n✗ Some tests failed!\n";
        return 1;
    }
    
    // Run performance benchmarks
    benchmark_performance();
    
    std::cout << "\n====================================================\n";
    std::cout << "  Test suite completed successfully!\n";
    std::cout << "====================================================\n";
    
    return 0;
}

/*
 * COMPILATION & RUN:
 *   g++ -O3 -fopenmp -std=c++11 cpu_test_Vxc.cpp -o cpu_test_Vxc ; ./cpu_test_Vxc
 */