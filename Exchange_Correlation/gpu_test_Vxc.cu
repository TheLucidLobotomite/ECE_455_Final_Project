#include "gpu_Vxc.cu"
#include <cmath>
#include <iostream>
#include <fstream>

// Physical constants
const double BOHR_TO_ANGSTROM = 0.529177;
const double ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM;

// Fe atomic positions in crystal coordinates from QE input
const int NUM_ATOMS = 4;
const double fe_positions_crystal[NUM_ATOMS][3] = {
    {0.0, 0.0, 0.0},
    {0.5, 0.5, 0.0},
    {0.5, 0.0, 0.5},
    {0.0, 0.5, 0.5}
};

// Global storage for atomic positions in Cartesian coordinates (Bohr)
double fe_positions_cart[NUM_ATOMS][3];
double lattice_constant;  // celldm(1) in Bohr

/**
 * Simple test density: n(r) = |r|
 * where |r| is the distance from origin
 */
double model_density(const double* r) {
    double x = r[0];
    double y = r[1];
    double z = r[2];
    
    double dist = sqrt(x*x + y*y + z*z);
    
    // Add small offset to avoid zero density at origin
    return dist + 1e-10;
}

/**
 * Convert crystal coordinates to Cartesian
 * For cubic cell (ibrav=1): r_cart = a * r_crystal
 */
void crystal_to_cartesian(const double cryst[3], double cart[3], double a) {
    cart[0] = a * cryst[0];
    cart[1] = a * cryst[1];
    cart[2] = a * cryst[2];
}

/**
 * Main test bench
 */
int main() {
    std::cout << "========================================\n";
    std::cout << "Vxc Test Bench for Fe FCC Crystal\n";
    std::cout << "========================================\n\n";
    
    // Extract parameters from QE input
    lattice_constant = 6.767109;  // celldm(1) in Bohr
    std::cout << "Lattice constant: " << lattice_constant << " Bohr"
              << " (" << lattice_constant * BOHR_TO_ANGSTROM << " Angstrom)\n";
    
    // For cubic cell (ibrav=1), lattice vectors are:
    double a1[3] = {lattice_constant, 0.0, 0.0};
    double a2[3] = {0.0, lattice_constant, 0.0};
    double a3[3] = {0.0, 0.0, lattice_constant};
    
    std::cout << "Lattice vectors (Bohr):\n";
    std::cout << "  a1 = [" << a1[0] << ", " << a1[1] << ", " << a1[2] << "]\n";
    std::cout << "  a2 = [" << a2[0] << ", " << a2[1] << ", " << a2[2] << "]\n";
    std::cout << "  a3 = [" << a3[0] << ", " << a3[1] << ", " << a3[2] << "]\n\n";
    
    // Convert Fe atomic positions to Cartesian
    std::cout << "Fe atomic positions:\n";
    std::cout << "  Crystal coords -> Cartesian (Bohr)\n";
    for (int i = 0; i < NUM_ATOMS; i++) {
        crystal_to_cartesian(fe_positions_crystal[i], fe_positions_cart[i], 
                           lattice_constant);
        std::cout << "  Fe" << i+1 << ": ["
                  << fe_positions_crystal[i][0] << ", "
                  << fe_positions_crystal[i][1] << ", "
                  << fe_positions_crystal[i][2] << "] -> ["
                  << fe_positions_cart[i][0] << ", "
                  << fe_positions_cart[i][1] << ", "
                  << fe_positions_cart[i][2] << "]\n";
    }
    std::cout << "\n";
    
    // Set up grid - using modest resolution for testing
    // QE uses FFT grid based on ecutrho, here we use 64^3 for speed
    int Nx = 64, Ny = 64, Nz = 64;
    std::cout << "Grid dimensions: " << Nx << " x " << Ny << " x " << Nz 
              << " = " << Nx*Ny*Nz << " points\n";
    std::cout << "Grid spacing: " 
              << lattice_constant/Nx << " Bohr/point\n\n";
    
    // Initialize Vxc context
    std::cout << "Initializing GPU context...\n";
    VxcContext* ctx = vxc_init(a1, a2, a3, Nx, Ny, Nz);
    std::cout << "GPU memory allocated for " << ctx->capacity << " points\n\n";
    
    // Compute density and Vxc
    std::cout << "Computing density at grid points...\n";
    vxc_compute_density(ctx, model_density);
    
    std::cout << "Computing Vxc on GPU...\n";
    vxc_compute(ctx);
    std::cout << "Vxc computation complete!\n\n";
    
    // Analyze results
    double vxc_min = 1e10, vxc_max = -1e10, vxc_sum = 0.0;
    double n_min = 1e10, n_max = -1e10, n_sum = 0.0;
    
    for (size_t i = 0; i < ctx->capacity; i++) {
        double vxc = ctx->h_vxc_pinned[i];
        double n = ctx->h_n_pinned[i];
        
        if (vxc < vxc_min) vxc_min = vxc;
        if (vxc > vxc_max) vxc_max = vxc;
        vxc_sum += vxc;
        
        if (n < n_min) n_min = n;
        if (n > n_max) n_max = n;
        n_sum += n;
    }
    
    double vxc_avg = vxc_sum / ctx->capacity;
    double n_avg = n_sum / ctx->capacity;
    
    std::cout << "========================================\n";
    std::cout << "Results Summary (Rydberg units)\n";
    std::cout << "========================================\n";
    std::cout << "Density statistics (electrons/Bohr³):\n";
    std::cout << "  Min:     " << n_min << "\n";
    std::cout << "  Max:     " << n_max << "\n";
    std::cout << "  Average: " << n_avg << "\n\n";
    
    std::cout << "Vxc statistics (Rydberg):\n";
    std::cout << "  Min:     " << vxc_min << " Ry\n";
    std::cout << "  Max:     " << vxc_max << " Ry\n";
    std::cout << "  Average: " << vxc_avg << " Ry\n\n";
    
    // Sample some points
    std::cout << "Sample Vxc values at specific points:\n";
    int sample_points[][3] = {
        {0, 0, 0},      // Origin
        {Nx/2, Ny/2, Nz/2},  // Center
        {Nx/4, Ny/4, Nz/4},  // Quarter point
    };
    
    for (int i = 0; i < 3; i++) {
        size_t idx = vxc_get_linear_index(ctx, sample_points[i][0], 
                                          sample_points[i][1], 
                                          sample_points[i][2]);
        double* r = &ctx->h_positions[3*idx];
        std::cout << "  (" << sample_points[i][0] << "," 
                  << sample_points[i][1] << "," 
                  << sample_points[i][2] << ") -> r=["
                  << r[0] << ", " << r[1] << ", " << r[2] 
                  << "] Bohr: n=" << ctx->h_n_pinned[idx]
                  << ", Vxc=" << ctx->h_vxc_pinned[idx] << " Ry\n";
    }
    std::cout << "\n";
    
    // Write results to file
    std::cout << "Writing results to vxc_output.dat...\n";
    std::ofstream outfile("vxc_output.dat");
    outfile << "# Vxc calculation for Fe FCC crystal\n";
    outfile << "# Columns: i j k x(Bohr) y(Bohr) z(Bohr) n(e/Bohr³) Vxc(Ry)\n";
    
    for (int i = 0; i < Nx; i += Nx/8) {  // Sample every 8th point to keep file small
        for (int j = 0; j < Ny; j += Ny/8) {
            for (int k = 0; k < Nz; k += Nz/8) {
                size_t idx = vxc_get_linear_index(ctx, i, j, k);
                double* r = &ctx->h_positions[3*idx];
                outfile << i << " " << j << " " << k << " "
                       << r[0] << " " << r[1] << " " << r[2] << " "
                       << ctx->h_n_pinned[idx] << " "
                       << ctx->h_vxc_pinned[idx] << "\n";
            }
        }
    }
    outfile.close();
    std::cout << "Results written successfully!\n\n";
    
    // Cleanup
    vxc_cleanup(ctx);
    std::cout << "GPU memory freed. Test complete!\n";
    
    return 0;
}