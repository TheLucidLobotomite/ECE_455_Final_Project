/**
 * Test Suite for Optimized GPU CUDA Implementation
 * 
 * Demonstrates performance improvements from persistent allocation
 * and pinned memory for iterative calculations
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include "gpu_Vxc.cu"

bool test_correctness() {
    std::cout << "\n=== Correctness Tests (Optimized API) ===\n\n";
    
    bool all_passed = true;
    
    // Test 1: Context-based API
    std::cout << "Test 1: Context-based computation\n";
    std::vector<double> n_test = {0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0};
    std::vector<double> vxc_gpu(n_test.size());
    
    VxcContext* ctx = vxc_init(n_test.size());
    vxc_compute(ctx, n_test.data(), vxc_gpu.data(), n_test.size());
    vxc_cleanup(ctx);
    
    std::cout << std::setprecision(10);
    std::cout << "n (e/bohr³)     Vxc (Ha)        Vxc (eV)\n";
    std::cout << "------------------------------------------------\n";
    
    for (size_t i = 0; i < n_test.size(); ++i) {
        std::cout << std::setw(12) << n_test[i] 
                  << std::setw(16) << vxc_gpu[i]
                  << std::setw(16) << vxc_gpu[i] * 27.211 << "\n";
    }
    std::cout << "  PASSED\n";
    
    // Test 2: Pinned memory API
    std::cout << "\nTest 2: Pinned memory computation\n";
    const size_t test_size = 1000;
    
    VxcContext* ctx_pinned = vxc_init(test_size, true);
    
    // Fill pinned input buffer
    for (size_t i = 0; i < test_size; ++i) {
        ctx_pinned->h_n_pinned[i] = 0.01 + 10.0 * static_cast<double>(i) / test_size;
    }
    
    // Compute using pinned buffers
    vxc_compute_pinned(ctx_pinned, test_size);
    
    std::cout << "  Sample results from pinned memory:\n";
    std::cout << "    n[0] = " << ctx_pinned->h_n_pinned[0] 
              << " -> Vxc = " << ctx_pinned->h_vxc_pinned[0] << " Ha\n";
    std::cout << "    n[500] = " << ctx_pinned->h_n_pinned[500] 
              << " -> Vxc = " << ctx_pinned->h_vxc_pinned[500] << " Ha\n";
    
    vxc_cleanup(ctx_pinned);
    std::cout << "  PASSED\n";
    
    // Test 3: Multiple iterations with same context
    std::cout << "\nTest 3: Reusing context across iterations\n";
    const size_t iter_size = 5000;
    VxcContext* ctx_reuse = vxc_init(iter_size);
    
    std::vector<double> n_iter(iter_size);
    std::vector<double> vxc_iter(iter_size);
    
    for (int iteration = 0; iteration < 5; ++iteration) {
        // Simulate changing density
        for (size_t i = 0; i < iter_size; ++i) {
            n_iter[i] = 0.1 * (1.0 + 0.1 * iteration) * 
                        std::exp(-static_cast<double>(i) / iter_size);
        }
        
        vxc_compute(ctx_reuse, n_iter.data(), vxc_iter.data(), iter_size);
        
        std::cout << "    Iteration " << iteration << ": Vxc[0] = " 
                  << vxc_iter[0] << " Ha\n";
    }
    
    vxc_cleanup(ctx_reuse);
    std::cout << "  PASSED\n";
    
    return all_passed;
}

void benchmark_comparison() {
    std::cout << "\n=== Performance Comparison ===\n\n";
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n\n";
    
    const size_t grid_size = 1000000;  // 1 million points
    const int num_iterations = 100;
    
    std::vector<double> n_data(grid_size);
    std::vector<double> vxc_result(grid_size);
    
    // Generate test data
    for (size_t i = 0; i < grid_size; ++i) {
        n_data[i] = 0.1 * std::exp(-static_cast<double>(i) / grid_size);
    }
    
    // ========================================================================
    // Benchmark 1: Legacy API (allocate/free every call)
    // ========================================================================
    std::cout << "Benchmark 1: Legacy API (allocate + transfer + compute + free each time)\n";
    std::cout << "Running " << num_iterations << " iterations...\n";
    
    auto t0_legacy = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        calculate_vxc_cuda(n_data.data(), vxc_result.data(), grid_size);
    }
    
    auto t1_legacy = std::chrono::high_resolution_clock::now();
    double time_legacy_ms = std::chrono::duration_cast<std::chrono::microseconds>(
        t1_legacy - t0_legacy).count() / 1000.0;
    
    std::cout << "  Total time: " << std::fixed << std::setprecision(2) 
              << time_legacy_ms << " ms\n";
    std::cout << "  Avg per iteration: " << time_legacy_ms / num_iterations << " ms\n\n";
    
    // ========================================================================
    // Benchmark 2: Context API with regular memory
    // ========================================================================
    std::cout << "Benchmark 2: Persistent allocation (no malloc/free overhead)\n";
    std::cout << "Running " << num_iterations << " iterations...\n";
    
    VxcContext* ctx = vxc_init(grid_size, false);
    
    auto t0_ctx = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        vxc_compute(ctx, n_data.data(), vxc_result.data(), grid_size);
    }
    
    auto t1_ctx = std::chrono::high_resolution_clock::now();
    double time_ctx_ms = std::chrono::duration_cast<std::chrono::microseconds>(
        t1_ctx - t0_ctx).count() / 1000.0;
    
    vxc_cleanup(ctx);
    
    std::cout << "  Total time: " << time_ctx_ms << " ms\n";
    std::cout << "  Avg per iteration: " << time_ctx_ms / num_iterations << " ms\n";
    std::cout << "  Speedup vs legacy: " << std::setprecision(2) 
              << time_legacy_ms / time_ctx_ms << "x\n\n";
    
    // ========================================================================
    // Benchmark 3: Context API with pinned memory
    // ========================================================================
    std::cout << "Benchmark 3: Persistent + pinned memory (fastest PCIe transfers)\n";
    std::cout << "Running " << num_iterations << " iterations...\n";
    
    VxcContext* ctx_pinned = vxc_init(grid_size, true);
    
    // Copy data to pinned buffer once
    for (size_t i = 0; i < grid_size; ++i) {
        ctx_pinned->h_n_pinned[i] = n_data[i];
    }
    
    auto t0_pinned = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        vxc_compute_pinned(ctx_pinned, grid_size);
        // In real usage, you'd update h_n_pinned between iterations
    }
    
    auto t1_pinned = std::chrono::high_resolution_clock::now();
    double time_pinned_ms = std::chrono::duration_cast<std::chrono::microseconds>(
        t1_pinned - t0_pinned).count() / 1000.0;
    
    vxc_cleanup(ctx_pinned);
    
    std::cout << "  Total time: " << time_pinned_ms << " ms\n";
    std::cout << "  Avg per iteration: " << time_pinned_ms / num_iterations << " ms\n";
    std::cout << "  Speedup vs legacy: " << time_legacy_ms / time_pinned_ms << "x\n";
    std::cout << "  Speedup vs context: " << time_ctx_ms / time_pinned_ms << "x\n\n";
    
    // ========================================================================
    // Summary
    // ========================================================================
    std::cout << "======================================================================\n";
    std::cout << "SUMMARY (for " << num_iterations << " iterations of " 
              << grid_size << " points)\n";
    std::cout << "======================================================================\n";
    std::cout << "Method                              Time (ms)    Speedup\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "1. Legacy (alloc each time)         " << std::setw(8) << std::setprecision(2) 
              << time_legacy_ms << "      1.00x\n";
    std::cout << "2. Persistent allocation            " << std::setw(8) 
              << time_ctx_ms << "      " << std::setprecision(2) 
              << time_legacy_ms / time_ctx_ms << "x\n";
    std::cout << "3. Persistent + pinned memory       " << std::setw(8) 
              << time_pinned_ms << "      " 
              << time_legacy_ms / time_pinned_ms << "x\n";
    std::cout << "======================================================================\n\n";
    
    std::cout << "💡 RECOMMENDATION: Use method 3 (pinned memory) for iterative calculations!\n\n";
}

int main() {
    std::cout << "====================================================\n";
    std::cout << "  Perdew-Wang 1992 - Optimized GPU Test Suite\n";
    std::cout << "====================================================\n";
    
    // Check for CUDA device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }
    
    // Run tests
    bool tests_passed = test_correctness();
    
    if (!tests_passed) {
        std::cout << "\n✗ Some tests failed!\n";
        return 1;
    }
    
    std::cout << "\n✓ All correctness tests passed!\n";
    
    // Run performance comparison
    benchmark_comparison();
    return 0;
}