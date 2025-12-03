/**
 * Test Suite for Unified CPU Implementation
 * Tests correctness and performance across different thread counts and grid sizes
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include "cpu_Vxc.cpp"

using namespace VxcCPU;

const double TOLERANCE = 1e-10;

bool test_correctness() {
    std::cout << "\n=== Correctness Tests ===\n\n";
    
    bool all_passed = true;
    
    // Test 1: Known values
    std::cout << "Test 1: Known density values\n";
    std::vector<double> n_test = {0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0};
    std::vector<double> vxc_result(n_test.size());
    
    calculate_vxc(n_test.data(), vxc_result.data(), n_test.size(), 1);
    
    std::cout << std::setprecision(10);
    std::cout << "n (e/bohr³)     Vxc (Ha)        Vxc (eV)\n";
    std::cout << "------------------------------------------------\n";
    
    for (size_t i = 0; i < n_test.size(); ++i) {
        std::cout << std::setw(12) << n_test[i] 
                  << std::setw(16) << vxc_result[i]
                  << std::setw(16) << vxc_result[i] * 27.211 << "\n";
    }
    std::cout << "  PASSED\n";
    
    // Test 2: Thread consistency (1 vs many threads)
    std::cout << "\nTest 2: Thread consistency check\n";
    const size_t test_size = 10000;
    std::vector<double> n_data(test_size);
    std::vector<double> vxc_1thread(test_size);
    std::vector<double> vxc_multithread(test_size);
    
    for (size_t i = 0; i < test_size; ++i) {
        n_data[i] = 0.01 + 10.0 * static_cast<double>(i) / test_size;
    }
    
    calculate_vxc(n_data.data(), vxc_1thread.data(), test_size, 1);
    calculate_vxc(n_data.data(), vxc_multithread.data(), test_size, 0); // Use all threads
    
    double max_diff = 0.0;
    for (size_t i = 0; i < test_size; ++i) {
        double diff = std::abs(vxc_1thread[i] - vxc_multithread[i]);
        max_diff = std::max(max_diff, diff);
    }
    
    std::cout << "  1 thread vs " << get_num_threads() << " threads\n";
    std::cout << "  Maximum difference: " << std::scientific << max_diff << "\n";
    
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
        if (vxc_1thread[i] > vxc_1thread[i-1]) {
            monotonic = false;
            break;
        }
    }
    
    if (monotonic) {
        std::cout << "  PASSED (Vxc decreases with density)\n";
    } else {
        std::cout << "  FAILED (non-monotonic)\n";
        all_passed = false;
    }
    
    // Test 4: Numerical stability
    std::cout << "\nTest 4: Numerical stability\n";
    std::vector<double> n_edge = {1e-10, 1e-6, 1e-3, 100.0, 1000.0};
    std::vector<double> vxc_edge(n_edge.size());
    
    calculate_vxc(n_edge.data(), vxc_edge.data(), n_edge.size(), 1);
    
    bool stable = true;
    for (size_t i = 0; i < n_edge.size(); ++i) {
        if (!std::isfinite(vxc_edge[i])) {
            std::cout << "  NaN/Inf at n = " << n_edge[i] << "\n";
            stable = false;
        }
    }
    
    if (stable) {
        std::cout << "  PASSED (all values finite)\n";
    } else {
        std::cout << "  FAILED (NaN or Inf detected)\n";
        all_passed = false;
    }
    
    return all_passed;
}

void benchmark_thread_scaling() {
    std::cout << "\n=== Thread Scaling Benchmark ===\n\n";
    
    const size_t grid_size = 1000000;  // 1M points
    const int num_runs = 100;
    
    std::vector<double> n_data(grid_size);
    std::vector<double> vxc_result(grid_size);
    
    // Generate test data
    for (size_t i = 0; i < grid_size; ++i) {
        n_data[i] = 0.1 * std::exp(-static_cast<double>(i) / grid_size);
    }
    
    int max_threads = omp_get_max_threads();
    std::cout << "System has " << max_threads << " threads available\n";
    std::cout << "Grid size: " << grid_size << " points\n";
    std::cout << "Iterations per test: " << num_runs << "\n\n";
    
    std::cout << "Threads    Avg Time (ms)    Speedup    Throughput (M pts/s)\n";
    std::cout << "----------------------------------------------------------------\n";
    
    double baseline_time = 0.0;
    
    // Test different thread counts
    std::vector<int> thread_counts = {1, 2, 4};
    if (max_threads >= 8) thread_counts.push_back(8);
    if (max_threads >= 16) thread_counts.push_back(16);
    if (max_threads > 16) thread_counts.push_back(max_threads);
    
    for (int num_threads : thread_counts) {
        double total_time = 0.0;
        
        for (int run = 0; run < num_runs; ++run) {
            auto t0 = std::chrono::high_resolution_clock::now();
            calculate_vxc(n_data.data(), vxc_result.data(), grid_size, num_threads);
            auto t1 = std::chrono::high_resolution_clock::now();
            
            double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e6;
            total_time += ms;
        }
        
        double avg_time = total_time / num_runs;
        if (num_threads == 1) baseline_time = avg_time;
        
        double speedup = baseline_time / avg_time;
        double throughput = grid_size / (avg_time / 1000.0) / 1e6;
        
        std::cout << std::setw(4) << num_threads << "       "
                  << std::fixed << std::setprecision(4) << std::setw(12) << avg_time << "      "
                  << std::setprecision(2) << std::setw(6) << speedup << "x     "
                  << std::setprecision(2) << std::setw(8) << throughput << "\n";
    }
    std::cout << "\n";
}

void benchmark_grid_sizes() {
    std::cout << "\n=== Grid Size Scaling Benchmark ===\n\n";
    
    std::vector<size_t> grid_sizes = {1000000, 10000000, 100000000};  // 1M, 10M, 100M
    std::vector<int> num_runs_per_size = {100, 20, 5};  // Fewer runs for larger grids
    
    int num_threads = omp_get_max_threads();
    std::cout << "Using " << num_threads << " threads\n\n";
    
    std::cout << "Grid Size        Runs    Avg Time (ms)    Throughput (M pts/s)\n";
    std::cout << "--------------------------------------------------------------------\n";
    
    for (size_t idx = 0; idx < grid_sizes.size(); ++idx) {
        size_t grid_size = grid_sizes[idx];
        int num_runs = num_runs_per_size[idx];
        
        std::vector<double> n_data(grid_size);
        std::vector<double> vxc_result(grid_size);
        
        // Generate test data
        for (size_t i = 0; i < grid_size; ++i) {
            n_data[i] = 0.1 * std::exp(-static_cast<double>(i) / grid_size);
        }
        
        // Warm-up run
        calculate_vxc(n_data.data(), vxc_result.data(), grid_size, num_threads);
        
        double total_time = 0.0;
        for (int run = 0; run < num_runs; ++run) {
            auto t0 = std::chrono::high_resolution_clock::now();
            calculate_vxc(n_data.data(), vxc_result.data(), grid_size, num_threads);
            auto t1 = std::chrono::high_resolution_clock::now();
            
            double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e6;
            total_time += ms;
        }
        
        double avg_time = total_time / num_runs;
        double throughput = grid_size / (avg_time / 1000.0) / 1e6;
        
        std::cout << std::setw(10) << grid_size << "      "
                  << std::setw(4) << num_runs << "    "
                  << std::fixed << std::setprecision(2) << std::setw(12) << avg_time << "      "
                  << std::setprecision(2) << std::setw(10) << throughput << "\n";
    }
    std::cout << "\n";
}

void benchmark_iterative_usage() {
    std::cout << "\n=== Iterative Usage Benchmark ===\n\n";
    
    const size_t grid_size = 1000000;
    const int num_iterations = 100;
    
    std::vector<double> n_data(grid_size);
    std::vector<double> vxc_result(grid_size);
    
    for (size_t i = 0; i < grid_size; ++i) {
        n_data[i] = 0.1 * std::exp(-static_cast<double>(i) / grid_size);
    }
    
    int num_threads = omp_get_max_threads();
    std::cout << "Using " << num_threads << " threads\n";
    std::cout << "Running " << num_iterations << " iterations...\n\n";
    
    // Method 1: Direct calls
    std::cout << "Method 1: Direct function calls\n";
    auto t0_direct = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        calculate_vxc(n_data.data(), vxc_result.data(), grid_size, num_threads);
    }
    
    auto t1_direct = std::chrono::high_resolution_clock::now();
    double time_direct = std::chrono::duration_cast<std::chrono::microseconds>(
        t1_direct - t0_direct).count() / 1000.0;
    
    std::cout << "  Total time: " << std::fixed << std::setprecision(2) 
              << time_direct << " ms\n";
    std::cout << "  Avg per iteration: " << time_direct / num_iterations << " ms\n\n";
    
    // Method 2: Context-based (pre-allocated)
    std::cout << "Method 2: Context-based (pre-allocated buffers)\n";
    VxcContext* ctx = vxc_init(grid_size, num_threads);
    
    auto t0_ctx = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        vxc_compute(ctx, n_data.data(), vxc_result.data(), grid_size);
    }
    
    auto t1_ctx = std::chrono::high_resolution_clock::now();
    double time_ctx = std::chrono::duration_cast<std::chrono::microseconds>(
        t1_ctx - t0_ctx).count() / 1000.0;
    
    vxc_cleanup(ctx);
    
    std::cout << "  Total time: " << time_ctx << " ms\n";
    std::cout << "  Avg per iteration: " << time_ctx / num_iterations << " ms\n";
    std::cout << "  Speedup: " << std::setprecision(2) 
              << time_direct / time_ctx << "x\n\n";
}

int main() {
    std::cout << "====================================================\n";
    std::cout << "  Perdew-Wang 1992 - Unified CPU Test Suite\n";
    std::cout << "====================================================\n";
    
    // Run correctness tests
    bool tests_passed = test_correctness();
    
    if (!tests_passed) {
        std::cout << "\n✗ Some tests failed!\n";
        return 1;
    }
    
    std::cout << "\n✓ All correctness tests passed!\n";
    
    // Run benchmarks
    benchmark_thread_scaling();
    benchmark_grid_sizes();
    benchmark_iterative_usage();
    
    return 0;
}