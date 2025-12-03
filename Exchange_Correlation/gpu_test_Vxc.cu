/**
 * Test Suite for Optimized GPU CUDA Implementation
 * Tests correctness and grid size scaling performance with pinned memory
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include "gpu_Vxc.cu"

void test_correctness() {
    std::cout << "\n=== Correctness Test ===\n\n";
    
    std::vector<double> n_test = {0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0};
    std::vector<double> vxc_gpu(n_test.size());
    
    VxcContext* ctx = vxc_init(n_test.size(), true);
    
    // Copy to pinned memory
    for (size_t i = 0; i < n_test.size(); ++i) {
        ctx->h_n_pinned[i] = n_test[i];
    }
    
    vxc_compute_pinned(ctx, n_test.size());
    
    // Copy results
    for (size_t i = 0; i < n_test.size(); ++i) {
        vxc_gpu[i] = ctx->h_vxc_pinned[i];
    }
    
    vxc_cleanup(ctx);
    
    std::cout << std::setprecision(10);
    std::cout << "n (e/bohr³)     Vxc (Ha)        Vxc (eV)\n";
    std::cout << "------------------------------------------------\n";
    
    for (size_t i = 0; i < n_test.size(); ++i) {
        std::cout << std::setw(12) << n_test[i] 
                  << std::setw(16) << vxc_gpu[i]
                  << std::setw(16) << vxc_gpu[i] * 27.211 << "\n";
    }
    std::cout << "\n";
}

void benchmark_grid_sizes() {
    std::cout << "=== Grid Size Scaling Benchmark ===\n\n";
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n";
    
    std::vector<size_t> grid_sizes = {
        100, 1000, 10000, 100000, 
        1000000, 10000000, 100000000
    };
    std::vector<int> num_runs_per_size = {
        1000, 1000, 1000, 500, 
        100, 20, 5
    };
    
    std::cout << "\nGrid Size        Runs    Avg Time (ms)    Throughput (M pts/s)\n";
    std::cout << "--------------------------------------------------------------------\n";
    
    for (size_t idx = 0; idx < grid_sizes.size(); ++idx) {
        size_t grid_size = grid_sizes[idx];
        int num_runs = num_runs_per_size[idx];
        
        // Use pinned memory for best performance
        VxcContext* ctx = vxc_init(grid_size, true);
        
        // Generate test data in pinned buffer
        for (size_t i = 0; i < grid_size; ++i) {
            ctx->h_n_pinned[i] = 0.1 * std::exp(-static_cast<double>(i) / grid_size);
        }
        
        // Warm-up run
        vxc_compute_pinned(ctx, grid_size);
        
        double total_time = 0.0;
        for (int run = 0; run < num_runs; ++run) {
            auto t0 = std::chrono::high_resolution_clock::now();
            vxc_compute_pinned(ctx, grid_size);
            auto t1 = std::chrono::high_resolution_clock::now();
            
            double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e6;
            total_time += ms;
        }
        
        vxc_cleanup(ctx);
        
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
    std::cout << "  Perdew-Wang 1992 - GPU Test Suite\n";
    std::cout << "====================================================\n";
    
    // Check for CUDA device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }
    
    test_correctness();
    benchmark_grid_sizes();
    
    return 0;
}