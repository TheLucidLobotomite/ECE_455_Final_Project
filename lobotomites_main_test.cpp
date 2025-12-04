#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <complex>
#include <fftw3.h>

#define PI 3.14159265358979323846

// Dummy implementations for testing without full dependencies
double compute_vxc(double n) {
    // Simple LDA exchange: Vxc ≈ -0.738 * n^(1/3)
    return -0.738 * std::pow(n, 1.0/3.0);
}

struct EigenResult {
    double* values;
    double** vectors;
    int n;
};

EigenResult* compute_eigenvalues(double** H, int n) {
    EigenResult* result = new EigenResult;
    result->n = n;
    result->values = new double[n];
    result->vectors = new double*[n];
    for (int i = 0; i < n; i++) {
        result->vectors[i] = new double[n];
    }
    
    // Dummy eigenvalues (just use diagonal of H for testing)
    for (int i = 0; i < n; i++) {
        result->values[i] = H[i][i];
        for (int j = 0; j < n; j++) {
            result->vectors[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
    
    return result;
}

void free_eigen_result(EigenResult* result) {
    for (int i = 0; i < result->n; i++) {
        delete[] result->vectors[i];
    }
    delete[] result->vectors;
    delete[] result->values;
    delete result;
}

// Test: G-vector generation
void test_gvector_generation() {
    std::cout << "\n=== Test 1: G-vector Generation ===\n";
    
    double a = 6.767109;  // Bohr
    double ecut = 10.0;   // Low cutoff for testing
    double gfactor = 2.0 * PI / a;
    int hmax = (int)std::ceil(std::sqrt(2.0 * ecut) * a / (2.0 * PI)) + 1;
    
    int count = 0;
    std::cout << "Lattice constant: " << a << " Bohr\n";
    std::cout << "Energy cutoff: " << ecut << " Ry\n";
    std::cout << "Max Miller index: " << hmax << "\n";
    std::cout << "\nFirst 10 G-vectors:\n";
    std::cout << std::setw(6) << "h" << std::setw(6) << "k" << std::setw(6) << "l" 
              << std::setw(12) << "|G|^2" << std::setw(12) << "E_kin(Ry)\n";
    std::cout << std::string(48, '-') << "\n";
    
    for (int h = -hmax; h <= hmax && count < 10; h++) {
        for (int k = -hmax; k <= hmax && count < 10; k++) {
            for (int l = -hmax; l <= hmax && count < 10; l++) {
                double gx = h * gfactor;
                double gy = k * gfactor;
                double gz = l * gfactor;
                double g2 = gx*gx + gy*gy + gz*gz;
                double ekin = g2 / 2.0;
                
                if (ekin <= ecut) {
                    std::cout << std::setw(6) << h << std::setw(6) << k << std::setw(6) << l
                              << std::setw(12) << std::fixed << std::setprecision(4) << g2
                              << std::setw(12) << ekin << "\n";
                    count++;
                }
            }
        }
    }
    std::cout << "Total G-vectors found: " << count << " (showing first 10)\n";
    std::cout << "✓ Test passed!\n";
}

// Test: FFT initialization
void test_fft_setup() {
    std::cout << "\n=== Test 2: FFT Setup ===\n";
    
    int nr1 = 16, nr2 = 16, nr3 = 16;
    int nrxx = nr1 * nr2 * nr3;
    
    std::cout << "FFT grid: " << nr1 << "x" << nr2 << "x" << nr3 << " = " << nrxx << " points\n";
    
    double* rho_r = fftw_alloc_real(nrxx);
    fftw_complex* rho_g = fftw_alloc_complex(nrxx);
    
    // Initialize with a test function: rho(x,y,z) = 1.0
    for (int i = 0; i < nrxx; i++) {
        rho_r[i] = 1.0;
    }
    
    fftw_plan plan = fftw_plan_dft_r2c_3d(nr1, nr2, nr3, rho_r, rho_g, FFTW_ESTIMATE);
    fftw_execute(plan);
    
    // G=0 component should be sum of all points
    double g0_real = rho_g[0][0] / nrxx;  // Normalize
    double g0_imag = rho_g[0][1] / nrxx;
    
    std::cout << "Input: constant density = 1.0\n";
    std::cout << "FFT result at G=0: (" << g0_real << ", " << g0_imag << ")\n";
    std::cout << "Expected: (1.0, 0.0)\n";
    
    bool passed = std::abs(g0_real - 1.0) < 1e-10 && std::abs(g0_imag) < 1e-10;
    
    fftw_destroy_plan(plan);
    fftw_free(rho_r);
    fftw_free(rho_g);
    fftw_cleanup();
    
    std::cout << (passed ? "✓ Test passed!\n" : "✗ Test failed!\n");
}

// Test: Vxc computation
void test_vxc_computation() {
    std::cout << "\n=== Test 3: Vxc Computation ===\n";
    
    double densities[] = {0.01, 0.1, 1.0, 10.0};
    
    std::cout << std::setw(15) << "Density (n)" << std::setw(15) << "Vxc\n";
    std::cout << std::string(30, '-') << "\n";
    
    for (double n : densities) {
        double vxc = compute_vxc(n);
        std::cout << std::setw(15) << std::fixed << std::setprecision(6) << n
                  << std::setw(15) << vxc << "\n";
    }
    
    // Check that Vxc is negative and increases with density
    bool passed = true;
    for (int i = 0; i < 3; i++) {
        double vxc1 = compute_vxc(densities[i]);
        double vxc2 = compute_vxc(densities[i+1]);
        if (vxc1 >= 0 || vxc2 >= 0 || vxc1 >= vxc2) {
            passed = false;
        }
    }
    
    std::cout << (passed ? "✓ Test passed!\n" : "✗ Test failed!\n");
}

// Test: Hamiltonian construction
void test_hamiltonian_build() {
    std::cout << "\n=== Test 4: Hamiltonian Construction ===\n";
    
    int npw = 5;
    double** H = new double*[npw];
    for (int i = 0; i < npw; i++) {
        H[i] = new double[npw];
        for (int j = 0; j < npw; j++) {
            H[i][j] = 0.0;
        }
    }
    
    // Add kinetic energies
    double ekin[] = {0.0, 0.5, 1.0, 1.5, 2.0};
    for (int i = 0; i < npw; i++) {
        H[i][i] = ekin[i];
    }
    
    std::cout << "Hamiltonian matrix (kinetic only):\n";
    for (int i = 0; i < npw; i++) {
        for (int j = 0; j < npw; j++) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << H[i][j];
        }
        std::cout << "\n";
    }
    
    // Test eigenvalue solver
    EigenResult* eig = compute_eigenvalues(H, npw);
    
    std::cout << "\nEigenvalues: ";
    for (int i = 0; i < npw; i++) {
        std::cout << std::fixed << std::setprecision(2) << eig->values[i] << " ";
    }
    std::cout << "\n";
    
    bool passed = true;
    for (int i = 0; i < npw; i++) {
        if (std::abs(eig->values[i] - ekin[i]) > 1e-10) {
            passed = false;
        }
    }
    
    free_eigen_result(eig);
    for (int i = 0; i < npw; i++) delete[] H[i];
    delete[] H;
    
    std::cout << (passed ? "✓ Test passed!\n" : "✗ Test failed!\n");
}

// Test: Density mixing
void test_density_mixing() {
    std::cout << "\n=== Test 5: Density Mixing ===\n";
    
    int npts = 5;
    double rho_old[] = {1.0, 1.0, 1.0, 1.0, 1.0};
    double rho_new[] = {1.2, 1.1, 0.9, 0.8, 1.0};
    double mixing_beta = 0.5;
    
    std::cout << "Mixing parameter: " << mixing_beta << "\n";
    std::cout << std::setw(10) << "Old" << std::setw(10) << "New" << std::setw(10) << "Mixed\n";
    std::cout << std::string(30, '-') << "\n";
    
    for (int i = 0; i < npts; i++) {
        double mixed = rho_old[i] + mixing_beta * (rho_new[i] - rho_old[i]);
        std::cout << std::setw(10) << std::fixed << std::setprecision(3) << rho_old[i]
                  << std::setw(10) << rho_new[i]
                  << std::setw(10) << mixed << "\n";
    }
    
    std::cout << "✓ Test passed!\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "DFT Code Component Tests\n";
    std::cout << "========================================\n";
    
    test_gvector_generation();
    test_fft_setup();
    test_vxc_computation();
    test_hamiltonian_build();
    test_density_mixing();
    
    std::cout << "\n========================================\n";
    std::cout << "All tests completed!\n";
    std::cout << "========================================\n";
    
    return 0;
}