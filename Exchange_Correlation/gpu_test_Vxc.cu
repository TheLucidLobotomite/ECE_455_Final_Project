/**
 * Test Suite for GPU CUDA Implementation
 * 
 * Tests correctness and performance of Perdew-Wang 1992 CUDA implementation
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include "gpu_Vxc.cu"

const double TOLERANCE_GPU = 1e-9;
const double PI_CPU = 3.14159265358979323846;

// CPU reference implementation for comparison
double calculate_Vxc_cpu(double n) {
    const double Cx = 0.738558766382022;
    double n_safe = std::max(n, 1e-12);
    double Vx = -Cx * std::pow(n_safe, 1.0/3.0);
    
    const double rs = std::pow(3.0 / (4.0 * PI_CPU * n_safe), 1.0/3.0);
    
    // Simplified epsilon_c
    const double A = 0.031091;
    const double alpha1 = 0.21370;
    const double beta1 = 7.5957;
    const double beta2 = 3.5876;
    const double beta3 = 1.6382;
    const double beta4 = 0.49294;
    
    double rs_sqrt = std::sqrt(rs);
    double rs_3_2 = rs * rs_sqrt;
    double rs_2 = rs * rs;
    
    double Q1 = 2.0 * A * (beta1 * rs_sqrt + beta2 * rs + beta3 * rs_3_2 + beta4 * rs_2);
    double ec = -2.0 * A * (1.0 + alpha1 * rs) * std::log(1.0 + 1.0 / Q1);
    
    const double h = 1e-6;
    double rs_h = rs + h;
    double rs_h_sqrt = std::sqrt(rs_h);
    double rs_h_3_2 = rs_h * rs_h_sqrt;
    double rs_h_2 = rs_h * rs_h;
    double Q1_h = 2.0 * A * (beta1 * rs_h_sqrt + beta2 * rs_h + beta3 * rs_h_3_2 + beta4 * rs_h_2);
    double ec_h = -2.0 * A * (1.0 + alpha1 * rs_h) * std::log(1.0 + 1.0 / Q1_h);
    
    double d_ec_drs = (ec_h - ec) / h;
    double Vc = ec - (rs / 3.0) * d_ec_drs;
    
    return Vx + Vc;
}

bool test_correctness() {
    std::cout << "\n=== Correctness Tests ===\n\n";
    
    bool all_passed = true;
    
    // Test 1: GPU vs CPU consistency
    std::cout << "Test 1: GPU vs CPU consistency\n";
    std::vector<double> n_test = {0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0};
    std::vector<double> vxc_gpu(n_test.size());
    
    calculate_vxc_cuda(n_test.data(), vxc_gpu.data(), n_test.size());
    
    double max_diff = 0.0;
    std::cout << std::setprecision(10);
    
    std::cout << "n (e/bohr³)     GPU Vxc (Ha)    CPU Vxc (Ha)    Difference\n";
    std::cout << "----------------------------------------------------------------\n";
    
    for (size_t i = 0; i < n_test.size(); ++i) {
        double vxc_cpu = calculate_Vxc_cpu(n_test[i]);
        double diff = std::abs(vxc_gpu[i] - vxc_cpu);
        max_diff = std::max(max_diff, diff);
        
        std::cout << std::setw(12) << n_test[i] 
                  << std::setw(16) << vxc_gpu[i]
                  << std::setw(16) << vxc_cpu
                  << std::setw(16) << diff << "\n";
    }
    
    std::cout << "\n  Maximum difference: " << max_diff << "\n";
    if (max_diff < TOLERANCE_GPU) {
        std::cout << "  PASSED\n";
    } else {
        std::cout << "  FAILED (difference too large)\n";
        all_passed = false;
    }
    
    // Test 2: Large array consistency
    std::cout << "\nTest 2: Large array consistency\n";
    const size_t test_size = 10000;
    std::vector<double> n_large(test_size);
    std::vector<double> vxc_gpu_large(test_size);
    
    for (size_t i = 0; i < test_size; ++i) {
        n_large[i] = 0.01 + 10.0 * static_cast<double>(i) / test_size;
    }
    
    calculate_vxc_cuda(n_large.data(), vxc_gpu_large.data(), test_size);
    
    // Check against CPU for sample points
    max_diff = 0.0;
    for (size_t i = 0; i < test_size; i += 1000) {
        double vxc_cpu = calculate_Vxc_cpu(n_large[i]);
        double diff = std::abs(vxc_gpu_large[i] - vxc_cpu);
        max_diff = std::max(max_diff, diff);
    }
    
    std::cout << "  Tested " << test_size << " points (sampled every 1000)\n";
    std::cout << "  Maximum difference: " << max_diff << "\n";
    if (max_diff < TOLERANCE_GPU) {
        std::cout << "  PASSED\n";
    } else {
        std::cout << "  FAILED\n";
        all_passed = false;
    }
    
    // Test 3: Monotonicity
    std::cout << "\nTest 3: Monotonicity check\n";
    bool monotonic = true;
    for (size_t i = 1; i < test_size; ++i) {
        if (vxc_gpu_large[i] > vxc_gpu_large[i-1]) {
            monotonic = false;
            break;
        }
    }
    
    if (monotonic) {
        std::cout << "  PASSED (Vxc monotonically decreases)\n";
    } else {
        std::cout << "  FAILED (non-monotonic)\n";
        all_passed = false;
    }
    
    // Test 4: No NaN or Inf
    std::cout << "\nTest 4: Numerical stability\n";
    bool stable = true;
    for (size_t i = 0; i < test_size; ++i) {
        if (!std::isfinite(vxc_gpu_large[i])) {
            stable = false;
            std::cout << "  NaN/Inf at index " << i << ", n = " << n_large[i] << "\n";
            break;
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

void benchmark_performance() {
    std::cout << "\n=== Performance Benchmarks ===\n\n";
    
    // Get GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n\n";
    
    std::vector<size_t> sizes = {100000, 1000000, 10000000, 50000000};
    
    for (size_t size : sizes) {
        std::cout << "Grid size: " << size << " points\n";
        
        std::vector<double> n_data(size);
        std::vector<double> vxc_result(size);
        
        for (size_t i = 0; i < size; ++i) {
            n_data[i] = 0.1 * std::exp(-static_cast<double>(i) / size);
        }
        
        // Warm-up
        calculate_vxc_cuda(n_data.data(), vxc_result.data(), size);
        
        // Timed run
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        calculate_vxc_cuda(n_data.data(), vxc_result.data(), size);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        
        std::cout << "  GPU time: " << std::fixed << std::setprecision(2) 
                  << milliseconds << " ms\n";
        std::cout << "  Throughput: " << size / (milliseconds / 1000.0) / 1e6 
                  << " million points/sec\n\n";
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
}

int main() {
    std::cout << "====================================================\n";
    std::cout << "  Perdew-Wang 1992 LDA - GPU CUDA Test Suite\n";
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
    
    if (tests_passed) {
        std::cout << "\n✓ All correctness tests passed!\n";
    } else {
        std::cout << "\n✗ Some tests failed!\n";
        return 1;
    }
    
    // Run benchmarks
    benchmark_performance();
    
    std::cout << "\n====================================================\n";
    std::cout << "  Test suite completed successfully!\n";
    std::cout << "====================================================\n";
    
    return 0;
}

/*
 * COMPILATION & RUN:
 *   nvcc -O3 -std=c++11 gpu_test_Vxc.cu -o gpu_test_Vxc ; ./gpu_test_Vxc
 */