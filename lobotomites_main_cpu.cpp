#ifndef LOBOTOMITES_MAIN_CPU_H
#define LOBOTOMITES_MAIN_CPU_H

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <complex>
#include <chrono>
#include <fftw3.h>

#define PI 3.14159265358979323846

/**
 * Structure to hold a G-vector and its properties
 */
struct GVector {
    int h, k, l;           // Miller indices
    double gx, gy, gz;     // Cartesian components (2π/a * h, k, l for cubic)
    double g2;             // |G|² in (2π/a)²
    double ekin;           // Kinetic energy ℏ²|G|²/(2m) in Rydberg
};

// ============================================
// CPU EIGENSOLVER (LAPACK)
// ============================================
#include "cpu_eigen.cpp"

// Include other components
#include "cpu_Vxc.cpp"
#include "hartree_planewave_use.cpp"
#include "pseudopotentials.cpp"

/**
 * Main DFT context structure
 */
struct DFTContext {
    // Lattice parameters
    double a1[3], a2[3], a3[3];  // Real space lattice vectors (Bohr)
    double b1[3], b2[3], b3[3];  // Reciprocal lattice vectors (2π/a)
    double cell_volume;           // Unit cell volume (Bohr³)
    
    // Plane wave basis
    std::vector<GVector> gvectors;  // List of G-vectors within cutoff
    int npw;                         // Number of plane waves
    double ecut_ry;                  // Energy cutoff (Rydberg)
    
    // FFT grid
    int nr1, nr2, nr3;              // FFT grid dimensions
    int nrxx;                        // Total FFT grid points
    fftw_plan plan_fwd;             // Real → reciprocal FFT
    fftw_plan plan_bwd;             // Reciprocal → real FFT
    double* rho_r;                   // Density in real space
    fftw_complex* rho_g;            // Density in reciprocal space
    double* vxc_r;                   // Vxc in real space
    fftw_complex* vxc_g;            // Vxc in reciprocal space
    
    // Hartree potential
    std::vector<double> Ck_real;    // Wave function coefficients (real part)
    std::vector<double> Ck_imag;    // Wave function coefficients (imag part)
    fftw_complex* vhart_g;          // Hartree potential in G-space
    
    // Pseudopotentials
    UPF* pseudopot;                 // Pseudopotential data
    bool use_pseudopot;             // Flag to enable/disable pseudopotentials
    double** Vnl_matrix;            // PRE-COMPUTED pseudopotential matrix (density independent!)
    
    // Wave functions
    std::vector<std::vector<std::complex<double>>> psi;  // Wave functions psi[band][G]
    std::vector<double> eigenvalues;                      // Eigenvalues (energies)
    int nbnd;                                             // Number of bands
    int nelec;                                            // Number of electrons
    
    // SCF parameters
    double mixing_beta;             // Density mixing parameter
    double conv_thr;                // Energy convergence threshold
    int max_iter;                   // Maximum SCF iterations
};

/**
 * Initialize reciprocal lattice vectors
 */
void init_reciprocal_lattice(DFTContext* ctx) {
    double a = ctx->a1[0];
    
    ctx->b1[0] = 2.0 * PI / a; ctx->b1[1] = 0.0;           ctx->b1[2] = 0.0;
    ctx->b2[0] = 0.0;           ctx->b2[1] = 2.0 * PI / a; ctx->b2[2] = 0.0;
    ctx->b3[0] = 0.0;           ctx->b3[1] = 0.0;           ctx->b3[2] = 2.0 * PI / a;
    
    ctx->cell_volume = a * a * a;
}

/**
 * Generate G-vectors within energy cutoff
 */
void generate_gvectors(DFTContext* ctx) {
    ctx->gvectors.clear();
    
    double a = ctx->a1[0];
    double gfactor = 2.0 * PI / a;
    
    int hmax = (int)std::ceil(std::sqrt(2.0 * ctx->ecut_ry) * a / (2.0 * PI)) + 1;
    
    std::cout << "Generating G-vectors with cutoff " << ctx->ecut_ry << " Ry\n";
    std::cout << "Searching Miller indices up to ±" << hmax << "\n";
    
    for (int h = -hmax; h <= hmax; h++) {
        for (int k = -hmax; k <= hmax; k++) {
            for (int l = -hmax; l <= hmax; l++) {
                GVector g;
                g.h = h; g.k = k; g.l = l;
                
                g.gx = h * gfactor;
                g.gy = k * gfactor;
                g.gz = l * gfactor;
                
                g.g2 = g.gx*g.gx + g.gy*g.gy + g.gz*g.gz;
                g.ekin = g.g2 / 2.0;
                
                if (g.ekin <= ctx->ecut_ry) {
                    ctx->gvectors.push_back(g);
                }
            }
        }
    }
    
    ctx->npw = ctx->gvectors.size();
    std::cout << "Number of plane waves: " << ctx->npw << "\n";
    std::cout << "Using CPU eigensolve (LAPACK)\n\n";
}

/**
 * Setup FFT grid
 */
void setup_fft_grid(DFTContext* ctx) {
    double a = ctx->a1[0];
    int nr_min = (int)std::ceil(2.0 * std::sqrt(ctx->ecut_ry) * a / PI);
    
    auto next_power_of_2 = [](int n) {
        int p = 1;
        while (p < n) p *= 2;
        return p;
    };
    
    ctx->nr1 = next_power_of_2(nr_min);
    ctx->nr2 = ctx->nr1;
    ctx->nr3 = ctx->nr1;
    ctx->nrxx = ctx->nr1 * ctx->nr2 * ctx->nr3;
    
    std::cout << "FFT grid: " << ctx->nr1 << " x " << ctx->nr2 << " x " << ctx->nr3 
              << " = " << ctx->nrxx << " points\n";
    
    ctx->rho_r = fftw_alloc_real(ctx->nrxx);
    ctx->rho_g = fftw_alloc_complex(ctx->nrxx);
    ctx->vxc_r = fftw_alloc_real(ctx->nrxx);
    ctx->vxc_g = fftw_alloc_complex(ctx->nrxx);
    ctx->vhart_g = fftw_alloc_complex(ctx->nrxx);
    
    ctx->Ck_real.resize(ctx->nrxx, 0.0);
    ctx->Ck_imag.resize(ctx->nrxx, 0.0);
    
    ctx->plan_fwd = fftw_plan_dft_r2c_3d(ctx->nr1, ctx->nr2, ctx->nr3,
                                         ctx->rho_r, ctx->rho_g,
                                         FFTW_ESTIMATE);
    ctx->plan_bwd = fftw_plan_dft_c2r_3d(ctx->nr1, ctx->nr2, ctx->nr3,
                                         ctx->rho_g, ctx->rho_r,
                                         FFTW_ESTIMATE);
    
    std::cout << "FFT plans created\n\n";
}

/**
 * Pre-compute pseudopotential matrix (ONCE before SCF loop)
 */
void setup_pseudopotential(DFTContext* ctx) {
    if (ctx->use_pseudopot && ctx->pseudopot != nullptr) {
        std::cout << "========================================\n";
        std::cout << "Pre-computing Pseudopotential Matrix\n";
        std::cout << "========================================\n";
        std::cout << "Note: This is computed ONCE (density-independent)\n";
        std::cout << "      and reused in all SCF iterations.\n\n";
        
        ctx->Vnl_matrix = precompute_pseudopotential_matrix(*ctx->pseudopot, ctx->gvectors, ctx->npw);
    } else {
        ctx->Vnl_matrix = nullptr;
        if (ctx->use_pseudopot && ctx->pseudopot == nullptr) {
            std::cout << "⚠ Warning: Pseudopotential enabled but not loaded.\n";
            std::cout << "  Continuing without pseudopotentials.\n\n";
            ctx->use_pseudopot = false;
        }
    }
}

/**
 * Initialize density
 */
void init_density(DFTContext* ctx) {
    double rho0 = ctx->nelec / ctx->cell_volume;
    
    for (int i = 0; i < ctx->nrxx; i++) {
        ctx->rho_r[i] = rho0;
    }
    
    std::cout << "Initial uniform density: " << std::scientific << std::setprecision(6) << rho0 << " electrons/Bohr³\n\n";
}

/**
 * Compute Vxc in real space
 */
void compute_vxc_realspace(DFTContext* ctx) {
    for (int i = 0; i < ctx->nrxx; i++) {
        double n = std::fmax(ctx->rho_r[i], 1e-12);
        ctx->vxc_r[i] = compute_vxc(n);
    }
}

/**
 * Transform Vxc from real space to reciprocal space
 */
void vxc_r2g(DFTContext* ctx) {
    for (int i = 0; i < ctx->nrxx; i++) {
        ctx->rho_r[i] = ctx->vxc_r[i];
    }
    
    fftw_execute(ctx->plan_fwd);
    
    for (int i = 0; i < ctx->nrxx; i++) {
        ctx->vxc_g[i][0] = ctx->rho_g[i][0] / ctx->nrxx;
        ctx->vxc_g[i][1] = ctx->rho_g[i][1] / ctx->nrxx;
    }
}

/**
 * Compute Hartree potential
 */
void compute_vhartree(DFTContext* ctx) {
    using namespace numint;
    
    for (int i = 0; i < ctx->nrxx; i++) {
        ctx->rho_r[i] = std::sqrt(std::fmax(ctx->rho_r[i], 0.0));
    }
    
    fftw_execute(ctx->plan_fwd);
    
    for (int i = 0; i < ctx->nrxx; i++) {
        ctx->Ck_real[i] = ctx->rho_g[i][0];
        ctx->Ck_imag[i] = ctx->rho_g[i][1];
    }
    
    double Lx = ctx->a1[0];
    double Ly = ctx->a2[1];
    double Lz = ctx->a3[2];
    
    TimedResult vh_center = Vh_PlaneWave_3D_p(
        ctx->Ck_real, ctx->Ck_imag,
        Lx, Ly, Lz,
        ctx->nr1, ctx->nr2, ctx->nr3,
        ctx->nr1/2, ctx->nr2/2, ctx->nr3/2
    );
    
    for (int i = 0; i < ctx->nrxx; i++) {
        ctx->vhart_g[i][0] = vh_center.value / ctx->nrxx;
        ctx->vhart_g[i][1] = 0.0;
    }
}

/**
 * Build Hamiltonian matrix
 */
void build_hamiltonian(DFTContext* ctx, double** H) {
    int npw = ctx->npw;
    
    auto t_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < npw; i++) {
        for (int j = 0; j < npw; j++) {
            H[i][j] = 0.0;
        }
    }
    
    // Kinetic energy
    for (int i = 0; i < npw; i++) {
        H[i][i] = ctx->gvectors[i].ekin;
    }
    
    auto t_kin = std::chrono::high_resolution_clock::now();
    double time_kin = std::chrono::duration<double>(t_kin - t_start).count();
    std::cout << "    Kinetic energy: " << std::fixed << std::setprecision(6) << time_kin << " s\n";
    
    // Vxc potential
    auto t_vxc_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < npw; i++) {
        int h = ctx->gvectors[i].h;
        int k = ctx->gvectors[i].k;
        int l = ctx->gvectors[i].l;
        
        int ih = (h >= 0) ? h : h + ctx->nr1;
        int ik = (k >= 0) ? k : k + ctx->nr2;
        int il = (l >= 0) ? l : l + ctx->nr3;
        
        if (ih >= 0 && ih < ctx->nr1 && ik >= 0 && ik < ctx->nr2 && il >= 0 && il < ctx->nr3) {
            int idx = ih * (ctx->nr2 * ctx->nr3) + ik * ctx->nr3 + il;
            H[i][i] += ctx->vxc_g[idx][0];
        }
    }
    auto t_vxc_end = std::chrono::high_resolution_clock::now();
    double time_vxc = std::chrono::duration<double>(t_vxc_end - t_vxc_start).count();
    std::cout << "    Vxc contribution: " << std::fixed << std::setprecision(6) << time_vxc << " s\n";
    
    // Hartree potential
    auto t_hartree_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < npw; i++) {
        int h = ctx->gvectors[i].h;
        int k = ctx->gvectors[i].k;
        int l = ctx->gvectors[i].l;
        
        int ih = (h >= 0) ? h : h + ctx->nr1;
        int ik = (k >= 0) ? k : k + ctx->nr2;
        int il = (l >= 0) ? l : l + ctx->nr3;
        
        if (ih >= 0 && ih < ctx->nr1 && ik >= 0 && ik < ctx->nr2 && il >= 0 && il < ctx->nr3) {
            int idx = ih * (ctx->nr2 * ctx->nr3) + ik * ctx->nr3 + il;
            H[i][i] += ctx->vhart_g[idx][0];
        }
    }
    auto t_hartree_end = std::chrono::high_resolution_clock::now();
    double time_hartree = std::chrono::duration<double>(t_hartree_end - t_hartree_start).count();
    std::cout << "    Hartree contribution: " << std::fixed << std::setprecision(6) << time_hartree << " s\n";
    
    // Pseudopotential
    if (ctx->use_pseudopot && ctx->Vnl_matrix != nullptr) {
        auto t_pseudo_start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < npw; i++) {
            for (int j = 0; j < npw; j++) {
                H[i][j] += ctx->Vnl_matrix[i][j];
            }
        }
        
        auto t_pseudo_end = std::chrono::high_resolution_clock::now();
        double time_pseudo = std::chrono::duration<double>(t_pseudo_end - t_pseudo_start).count();
        std::cout << "    Pseudopotential contribution (pre-computed): " << std::fixed << std::setprecision(6) << time_pseudo << " s\n";
    }
}

/**
 * Compute new density from wave functions
 */
void compute_density_from_wfn(DFTContext* ctx) {
    for (int i = 0; i < ctx->nrxx; i++) {
        ctx->rho_r[i] = 0.0;
    }
    
    int nocc = ctx->nelec / 2;
    
    for (int ibnd = 0; ibnd < nocc; ibnd++) {
        for (int i = 0; i < ctx->nrxx; i++) {
            ctx->rho_g[i][0] = 0.0;
            ctx->rho_g[i][1] = 0.0;
        }
        
        for (int ig = 0; ig < ctx->npw; ig++) {
            int h = ctx->gvectors[ig].h;
            int k = ctx->gvectors[ig].k;
            int l = ctx->gvectors[ig].l;
            
            int ih = (h >= 0) ? h : h + ctx->nr1;
            int ik = (k >= 0) ? k : k + ctx->nr2;
            int il = (l >= 0) ? l : l + ctx->nr3;
            
            if (ih >= 0 && ih < ctx->nr1 && ik >= 0 && ik < ctx->nr2 && il >= 0 && il < ctx->nr3) {
                int idx = ih * (ctx->nr2 * ctx->nr3) + ik * ctx->nr3 + il;
                ctx->rho_g[idx][0] = ctx->psi[ibnd][ig].real();
                ctx->rho_g[idx][1] = ctx->psi[ibnd][ig].imag();
            }
        }
        
        fftw_execute(ctx->plan_bwd);
        
        double occ = 2.0;
        for (int ir = 0; ir < ctx->nrxx; ir++) {
            double psi_r = ctx->rho_r[ir] / ctx->nrxx;
            ctx->rho_r[ir] += occ * psi_r * psi_r;
        }
    }
}

/**
 * Mix densities
 */
void mix_density(DFTContext* ctx, double* rho_in, double* rho_new) {
    for (int i = 0; i < ctx->nrxx; i++) {
        rho_new[i] = rho_in[i] + ctx->mixing_beta * (ctx->rho_r[i] - rho_in[i]);
        rho_in[i] = rho_new[i];
    }
}

/**
 * Compute total energy
 */
double compute_total_energy(DFTContext* ctx) {
    double e_kin = 0.0;
    double e_xc = 0.0;
    double e_hartree = 0.0;
    
    int nocc = ctx->nelec / 2;
    for (int i = 0; i < nocc; i++) {
        e_kin += 2.0 * ctx->eigenvalues[i];
    }
    
    double dv = ctx->cell_volume / ctx->nrxx;
    for (int i = 0; i < ctx->nrxx; i++) {
        double n = ctx->rho_r[i];
        double vxc = ctx->vxc_r[i];
        e_xc += n * vxc * dv;
    }
    
    for (int i = 0; i < ctx->nrxx; i++) {
        e_hartree += 0.5 * ctx->rho_r[i] * ctx->vhart_g[0][0] * dv;
    }
    
    return e_kin + e_xc + e_hartree;
}

/**
 * Main SCF loop
 */
void run_scf(DFTContext* ctx) {
    std::cout << "\n========================================\n";
    std::cout << "Starting Self-Consistent Field Loop\n";
    std::cout << "========================================\n\n";
    
    double e_old = 0.0;
    double* rho_in = new double[ctx->nrxx];
    
    for (int i = 0; i < ctx->nrxx; i++) {
        rho_in[i] = ctx->rho_r[i];
    }
    
    for (int iter = 1; iter <= ctx->max_iter; iter++) {
        std::cout << "SCF Iteration " << iter << "\n";
        std::cout << std::string(50, '-') << "\n";
        
        std::cout << "  Computing Vxc...\n";
        auto t_vxc_step_start = std::chrono::high_resolution_clock::now();
        compute_vxc_realspace(ctx);
        vxc_r2g(ctx);
        auto t_vxc_step_end = std::chrono::high_resolution_clock::now();
        double time_vxc_step = std::chrono::duration<double>(t_vxc_step_end - t_vxc_step_start).count();
        std::cout << "    Total Vxc step time: " << std::fixed << std::setprecision(6) << time_vxc_step << " s\n";
        
        std::cout << "  Computing Hartree potential...\n";
        auto t_hartree_step_start = std::chrono::high_resolution_clock::now();
        compute_vhartree(ctx);
        auto t_hartree_step_end = std::chrono::high_resolution_clock::now();
        double time_hartree_step = std::chrono::duration<double>(t_hartree_step_end - t_hartree_step_start).count();
        std::cout << "    Total Hartree step time: " << std::fixed << std::setprecision(6) << time_hartree_step << " s\n";
        
        std::cout << "  Building Hamiltonian...\n";
        auto t_build_start = std::chrono::high_resolution_clock::now();
        double** H = new double*[ctx->npw];
        for (int i = 0; i < ctx->npw; i++) {
            H[i] = new double[ctx->npw];
        }
        build_hamiltonian(ctx, H);
        auto t_build_end = std::chrono::high_resolution_clock::now();
        double time_build = std::chrono::duration<double>(t_build_end - t_build_start).count();
        std::cout << "    Total Hamiltonian build time: " << std::fixed << std::setprecision(6) << time_build << " s\n";
        
        std::cout << "  Diagonalizing (" << ctx->npw << "x" << ctx->npw << ")...\n";
        auto t_diag_start = std::chrono::high_resolution_clock::now();
        EigenResult* eig = compute_eigenvalues(H, ctx->npw);
        auto t_diag_end = std::chrono::high_resolution_clock::now();
        double time_diag = std::chrono::duration<double>(t_diag_end - t_diag_start).count();
        std::cout << "    Diagonalization time: " << std::fixed << std::setprecision(6) << time_diag << " s\n";
        
        ctx->eigenvalues.resize(ctx->nbnd);
        ctx->psi.resize(ctx->nbnd);
        for (int i = 0; i < ctx->nbnd; i++) {
            ctx->eigenvalues[i] = eig->values[i];
            ctx->psi[i].resize(ctx->npw);
            for (int j = 0; j < ctx->npw; j++) {
                ctx->psi[i][j] = std::complex<double>(eig->vectors[i][j], 0.0);
            }
        }
        
        std::cout << "  Lowest eigenvalues (Ry): ";
        for (int i = 0; i < std::min(5, ctx->nbnd); i++) {
            std::cout << std::fixed << std::setprecision(4) << ctx->eigenvalues[i] << " ";
        }
        std::cout << "\n";
        
        std::cout << "  Computing new density...\n";
        auto t_density_start = std::chrono::high_resolution_clock::now();
        compute_density_from_wfn(ctx);
        auto t_density_end = std::chrono::high_resolution_clock::now();
        double time_density = std::chrono::duration<double>(t_density_end - t_density_start).count();
        std::cout << "    Density computation time: " << std::fixed << std::setprecision(6) << time_density << " s\n";
        
        auto t_mix_start = std::chrono::high_resolution_clock::now();
        mix_density(ctx, rho_in, ctx->rho_r);
        auto t_mix_end = std::chrono::high_resolution_clock::now();
        double time_mix = std::chrono::duration<double>(t_mix_end - t_mix_start).count();
        std::cout << "    Density mixing time: " << std::fixed << std::setprecision(6) << time_mix << " s\n";
        
        double e_total = compute_total_energy(ctx);
        double de = std::abs(e_total - e_old);
        
        std::cout << "  Total Energy: " << std::fixed << std::setprecision(8) 
                  << e_total << " Ry\n";
        std::cout << "  |ΔE|:         " << std::scientific << std::setprecision(3) 
                  << de << " Ry\n";
        
        if (iter > 1 && de < ctx->conv_thr) {
            std::cout << "\n✓ SCF CONVERGED in " << iter << " iterations!\n";
            std::cout << "  Final Energy: " << std::fixed << std::setprecision(8) 
                      << e_total << " Ry\n\n";
            
            free_eigen_result(eig);
            for (int i = 0; i < ctx->npw; i++) delete[] H[i];
            delete[] H;
            break;
        }
        
        e_old = e_total;
        
        free_eigen_result(eig);
        for (int i = 0; i < ctx->npw; i++) delete[] H[i];
        delete[] H;
        
        std::cout << "\n";
        
        if (iter == ctx->max_iter) {
            std::cout << "✗ SCF NOT CONVERGED after " << ctx->max_iter << " iterations\n\n";
        }
    }
    
    delete[] rho_in;
}

/**
 * Cleanup
 */
void cleanup_dft_context(DFTContext* ctx) {
    fftw_destroy_plan(ctx->plan_fwd);
    fftw_destroy_plan(ctx->plan_bwd);
    fftw_free(ctx->rho_r);
    fftw_free(ctx->rho_g);
    fftw_free(ctx->vxc_r);
    fftw_free(ctx->vxc_g);
    fftw_free(ctx->vhart_g);
    fftw_cleanup();
    
    if (ctx->Vnl_matrix != nullptr) {
        free_pseudopotential_matrix(ctx->Vnl_matrix, ctx->npw);
        ctx->Vnl_matrix = nullptr;
    }
}

#endif // LOBOTOMITES_MAIN_CPU_H