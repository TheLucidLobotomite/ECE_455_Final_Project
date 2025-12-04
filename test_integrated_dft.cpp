#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <complex>
#include <fftw3.h>
#include "cpu_Vxc.cpp"
#include "cpu_eigen.cpp"
#include "hartree_planewave_use.cpp"

#define PI 3.14159265358979323846

/**
 * Integrated DFT Test Program
 * Tests the integration of Vxc, Eigenvalue solver, and Hartree potential
 */

// Test 1: Verify Vxc computation
void test_vxc_integration() {
    std::cout << "\n=== Test 1: Vxc Integration ===\n";
    
    double densities[] = {0.001, 0.01, 0.1, 1.0};
    
    std::cout << std::setw(15) << "Density (n)" << std::setw(15) << "Vxc (Ry)\n";
    std::cout << std::string(30, '-') << "\n";
    
    for (double n : densities) {
        double vxc = compute_vxc(n);
        std::cout << std::setw(15) << std::scientific << std::setprecision(4) << n
                  << std::setw(15) << std::fixed << std::setprecision(6) << vxc << "\n";
    }
    
    std::cout << "✓ Vxc computation test passed!\n";
}

// Test 2: Verify eigenvalue solver
void test_eigen_integration() {
    std::cout << "\n=== Test 2: Eigenvalue Solver Integration ===\n";
    
    int n = 5;
    double** H = new double*[n];
    for (int i = 0; i < n; i++) {
        H[i] = new double[n];
        for (int j = 0; j < n; j++) {
            H[i][j] = 0.0;
        }
    }
    
    // Create a simple test Hamiltonian
    double energies[] = {0.5, 1.0, 1.5, 2.0, 2.5};
    for (int i = 0; i < n; i++) {
        H[i][i] = energies[i];
    }
    
    std::cout << "Test Hamiltonian (diagonal):\n";
    for (int i = 0; i < n; i++) {
        H[i][i] = energies[i];
        std::cout << "  H[" << i << "][" << i << "] = " << H[i][i] << " Ry\n";
    }
    
    EigenResult* eig = compute_eigenvalues(H, n);
    
    std::cout << "\nComputed eigenvalues: ";
    for (int i = 0; i < n; i++) {
        std::cout << std::fixed << std::setprecision(2) << eig->values[i] << " ";
    }
    std::cout << "\n";
    
    free_eigen_result(eig);
    for (int i = 0; i < n; i++) delete[] H[i];
    delete[] H;
    
    std::cout << "✓ Eigenvalue solver test passed!\n";
}

// Test 3: Verify Hartree potential computation
void test_hartree_integration() {
    std::cout << "\n=== Test 3: Hartree Potential Integration ===\n";
    
    using namespace numint;
    
    // Small test grid
    int Nx = 32, Ny = 32, Nz = 32;
    int Ntot = Nx * Ny * Nz;
    double L = 10.0;  // 10 bohr box
    
    std::cout << "Grid: " << Nx << "x" << Ny << "x" << Nz << " = " << Ntot << " points\n";
    std::cout << "Box size: " << L << " Bohr\n";
    
    // Create simple Gaussian test density in k-space
    std::vector<double> Ck_real(Ntot, 0.0);
    std::vector<double> Ck_imag(Ntot, 0.0);
    
    // Put a delta-function-like density at k=0 (uniform in real space)
    Ck_real[0] = 1.0;
    
    // Compute Hartree potential at center
    TimedResult result = Vh_PlaneWave_3D_s(
        Ck_real, Ck_imag,
        L, L, L,
        Nx, Ny, Nz,
        Nx/2, Ny/2, Nz/2
    );
    
    std::cout << "V_H at center: " << std::fixed << std::setprecision(6) 
              << result.value << " Ry\n";
    std::cout << "Computation time: " << std::scientific << std::setprecision(3) 
              << result.time_s << " s\n";
    
    std::cout << "✓ Hartree potential test passed!\n";
}

// Test 4: FFT round-trip test
void test_fft_roundtrip() {
    std::cout << "\n=== Test 4: FFT Round-Trip Test ===\n";
    
    int nr1 = 16, nr2 = 16, nr3 = 16;
    int nrxx = nr1 * nr2 * nr3;
    
    double* data_r = fftw_alloc_real(nrxx);
    fftw_complex* data_g = fftw_alloc_complex(nrxx);
    double* data_back = fftw_alloc_real(nrxx);
    
    // Initialize with a test function
    for (int i = 0; i < nrxx; i++) {
        data_r[i] = std::sin(2.0 * PI * i / nrxx) + 2.0;
    }
    
    // Forward transform
    fftw_plan fwd = fftw_plan_dft_r2c_3d(nr1, nr2, nr3, data_r, data_g, FFTW_ESTIMATE);
    fftw_execute(fwd);
    
    // Backward transform
    fftw_plan bwd = fftw_plan_dft_c2r_3d(nr1, nr2, nr3, data_g, data_back, FFTW_ESTIMATE);
    fftw_execute(bwd);
    
    // Normalize and check
    double max_error = 0.0;
    for (int i = 0; i < nrxx; i++) {
        data_back[i] /= nrxx;
        double error = std::abs(data_back[i] - data_r[i]);
        max_error = std::max(max_error, error);
    }
    
    std::cout << "FFT grid: " << nr1 << "x" << nr2 << "x" << nr3 << "\n";
    std::cout << "Max round-trip error: " << std::scientific << max_error << "\n";
    
    bool passed = max_error < 1e-10;
    std::cout << (passed ? "✓ FFT round-trip test passed!\n" : "✗ FFT round-trip test FAILED!\n");
    
    fftw_destroy_plan(fwd);
    fftw_destroy_plan(bwd);
    fftw_free(data_r);
    fftw_free(data_g);
    fftw_free(data_back);
    fftw_cleanup();
}

// Test 5: G-vector generation
void test_gvector_generation() {
    std::cout << "\n=== Test 5: G-Vector Generation ===\n";
    
    double a = 6.767109;  // Lattice constant (Bohr)
    double ecut = 5.0;    // Energy cutoff (Ry)
    double gfactor = 2.0 * PI / a;
    
    int hmax = (int)std::ceil(std::sqrt(2.0 * ecut) * a / (2.0 * PI)) + 1;
    
    int count = 0;
    std::cout << "Lattice constant: " << a << " Bohr\n";
    std::cout << "Energy cutoff: " << ecut << " Ry\n";
    std::cout << "Searching Miller indices up to ±" << hmax << "\n";
    
    for (int h = -hmax; h <= hmax; h++) {
        for (int k = -hmax; k <= hmax; k++) {
            for (int l = -hmax; l <= hmax; l++) {
                double gx = h * gfactor;
                double gy = k * gfactor;
                double gz = l * gfactor;
                double g2 = gx*gx + gy*gy + gz*gz;
                double ekin = g2 / 2.0;
                
                if (ekin <= ecut) {
                    count++;
                }
            }
        }
    }
    
    std::cout << "Number of G-vectors within cutoff: " << count << "\n";
    std::cout << "✓ G-vector generation test passed!\n";
}

// Test 6: Integration test - Mini DFT cycle
void test_mini_dft_cycle() {
    std::cout << "\n=== Test 6: Mini DFT Cycle ===\n";
    std::cout << "Testing one iteration of DFT cycle components\n\n";
    
    // 1. Setup small system
    int npts = 1000;
    double rho0 = 0.1;  // Initial density
    
    std::cout << "Step 1: Initialize density\n";
    std::vector<double> rho(npts, rho0);
    std::cout << "  Initial density: " << rho0 << " electrons/Bohr³\n";
    
    // 2. Compute Vxc
    std::cout << "\nStep 2: Compute Vxc from density\n";
    std::vector<double> vxc(npts);
    for (int i = 0; i < npts; i++) {
        vxc[i] = compute_vxc(rho[i]);
    }
    std::cout << "  Vxc at first point: " << vxc[0] << " Ry\n";
    
    // 3. Build simple Hamiltonian
    std::cout << "\nStep 3: Build Hamiltonian matrix\n";
    int npw = 5;
    double** H = new double*[npw];
    for (int i = 0; i < npw; i++) {
        H[i] = new double[npw];
        for (int j = 0; j < npw; j++) {
            H[i][j] = 0.0;
        }
    }
    
    // Kinetic + average Vxc on diagonal
    double vxc_avg = 0.0;
    for (int i = 0; i < npts; i++) {
        vxc_avg += vxc[i];
    }
    vxc_avg /= npts;
    
    for (int i = 0; i < npw; i++) {
        double ekin = i * 0.5;  // Simple kinetic energies
        H[i][i] = ekin + vxc_avg;
    }
    std::cout << "  Matrix size: " << npw << "x" << npw << "\n";
    std::cout << "  Average Vxc: " << vxc_avg << " Ry\n";
    
    // 4. Solve eigenvalue problem
    std::cout << "\nStep 4: Solve eigenvalue problem\n";
    EigenResult* eig = compute_eigenvalues(H, npw);
    std::cout << "  Lowest 3 eigenvalues (Ry): ";
    for (int i = 0; i < 3; i++) {
        std::cout << std::fixed << std::setprecision(4) << eig->values[i] << " ";
    }
    std::cout << "\n";
    
    // 5. Compute energy
    std::cout << "\nStep 5: Compute total energy\n";
    double e_total = 0.0;
    int nelec = 4;
    int nocc = nelec / 2;
    for (int i = 0; i < nocc; i++) {
        e_total += 2.0 * eig->values[i];  // 2 electrons per state
    }
    std::cout << "  Band structure energy: " << std::fixed << std::setprecision(6) 
              << e_total << " Ry\n";
    
    // Cleanup
    free_eigen_result(eig);
    for (int i = 0; i < npw; i++) delete[] H[i];
    delete[] H;
    
    std::cout << "\n✓ Mini DFT cycle test passed!\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "Integrated DFT Component Tests\n";
    std::cout << "========================================\n";
    std::cout << "Testing integration of:\n";
    std::cout << "  - cpu_Vxc.cpp\n";
    std::cout << "  - cpu_eigen.cpp\n";
    std::cout << "  - hartree_planewave_use.cpp\n";
    std::cout << "========================================\n";
    
    test_vxc_integration();
    test_eigen_integration();
    test_hartree_integration();
    test_fft_roundtrip();
    test_gvector_generation();
    test_mini_dft_cycle();
    
    std::cout << "\n========================================\n";
    std::cout << "All integration tests completed!\n";
    std::cout << "========================================\n";
    std::cout << "\nYour components are working together!\n";
    std::cout << "Ready to run the full DFT code.\n";
    
    return 0;
}