/**
 * Test Suite for Unified CPU Implementation
 * Tests correctness and grid size scaling performance
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include "cpu_Vxc.cpp"

using namespace VxcCPU;

void test_correctness() {
    std::cout << "\n=== Correctness Test ===\n\n";
    
    std::vector<double> n_test = {0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0};
    std::vector<double> vxc_result(n_test.size());
    
    int num_threads = omp_get_max_threads();
    calculate_vxc(n_test.data(), vxc_result.data(), n_test.size(), num_threads);
    
    std::cout << std::setprecision(10);
    std::cout << "n (e/bohr³)     Vxc (Ha)        Vxc (eV)\n";
    std::cout << "------------------------------------------------\n";
    
    for (size_t i = 0; i < n_test.size(); ++i) {
        std::cout << std::setw(12) << n_test[i] 
                  << std::setw(16) << vxc_result[i]
                  << std::setw(16) << vxc_result[i] * 27.211 << "\n";
    }
    std::cout << "\n";
}

void benchmark_grid_sizes() {
    std::cout << "=== Grid Size Scaling Benchmark ===\n\n";
    
    std::vector<size_t> grid_sizes = {
        100, 1000, 10000, 100000, 
        1000000, 10000000, 100000000
    };
    std::vector<int> num_runs_per_size = {
        1000, 1000, 1000, 500, 
        100, 20, 5
    };
    
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);  // Ensure we use max threads
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

int main() {
    std::cout << "====================================================\n";
    std::cout << "  Perdew-Wang 1992 - CPU Test Suite\n";
    std::cout << "====================================================\n";
    
    test_correctness();
    benchmark_grid_sizes();
    
    return 0;
}