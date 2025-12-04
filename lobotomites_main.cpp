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
 * Structure to hold a G-vector and its properties
 */
struct GVector {
    int h, k, l;           // Miller indices
    double gx, gy, gz;     // Cartesian components (2π/a * h, k, l for cubic)
    double g2;             // |G|² in (2π/a)²
    double ekin;           // Kinetic energy ℏ²|G|²/(2m) in Rydberg
};

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
 * For simple cubic: b_i = 2π/a * e_i
 */
void init_reciprocal_lattice(DFTContext* ctx) {
    double a = ctx->a1[0];  // Lattice constant
    
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
    std::cout << "Number of plane waves: " << ctx->npw << "\n\n";
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
 * Initialize density - start with uniform density
 */
void init_density(DFTContext* ctx) {
    double rho0 = ctx->nelec / ctx->cell_volume;
    
    for (int i = 0; i < ctx->nrxx; i++) {
        ctx->rho_r[i] = rho0;
    }
    
    std::cout << "Initial uniform density: " << rho0 << " electrons/Bohr³\n\n";
}

/**
 * Compute Vxc in real space using cpu_Vxc.cpp
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
 * Compute Hartree potential using hartree_planewave_use.cpp
 */
void compute_vhartree(DFTContext* ctx) {
    using namespace numint;
    
    // Prepare wave function coefficients from current density
    // Transform density to k-space to get coefficients
    for (int i = 0; i < ctx->nrxx; i++) {
        ctx->rho_r[i] = std::sqrt(std::fmax(ctx->rho_r[i], 0.0));
    }
    
    fftw_execute(ctx->plan_fwd);
    
    for (int i = 0; i < ctx->nrxx; i++) {
        ctx->Ck_real[i] = ctx->rho_g[i][0];
        ctx->Ck_imag[i] = ctx->rho_g[i][1];
    }
    
    // Compute Hartree potential at each grid point and store in vhart_g
    // For efficiency, we compute at representative points and interpolate
    // Here we'll use a simplified approach: compute average Hartree potential
    
    double Lx = ctx->a1[0];
    double Ly = ctx->a2[1];
    double Lz = ctx->a3[2];
    
    // Compute at center point as representative value
    TimedResult vh_center = Vh_PlaneWave_3D_s(
        ctx->Ck_real, ctx->Ck_imag,
        Lx, Ly, Lz,
        ctx->nr1, ctx->nr2, ctx->nr3,
        ctx->nr1/2, ctx->nr2/2, ctx->nr3/2
    );
    
    // For simplicity, use uniform Hartree potential (diagonal approximation)
    // In a full implementation, you'd compute V_H(G) properly
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
    
    for (int i = 0; i < npw; i++) {
        for (int j = 0; j < npw; j++) {
            H[i][j] = 0.0;
        }
    }
    
    // Add kinetic energy (diagonal)
    for (int i = 0; i < npw; i++) {
        H[i][i] = ctx->gvectors[i].ekin;
    }
    
    // Add potential terms (diagonal approximation)
    for (int i = 0; i < npw; i++) {
        int h = ctx->gvectors[i].h;
        int k = ctx->gvectors[i].k;
        int l = ctx->gvectors[i].l;
        
        int ih = (h >= 0) ? h : h + ctx->nr1;
        int ik = (k >= 0) ? k : k + ctx->nr2;
        int il = (l >= 0) ? l : l + ctx->nr3;
        
        if (ih >= 0 && ih < ctx->nr1 && ik >= 0 && ik < ctx->nr2 && il >= 0 && il < ctx->nr3) {
            int idx = ih * (ctx->nr2 * ctx->nr3) + ik * ctx->nr3 + il;
            
            H[i][i] += ctx->vxc_g[idx][0] + ctx->vhart_g[idx][0];
        }
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
    
    // Hartree energy (simplified)
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
        
        // 1. Compute Vxc
        std::cout << "  Computing Vxc...\n";
        compute_vxc_realspace(ctx);
        vxc_r2g(ctx);
        
        // 2. Compute Hartree potential
        std::cout << "  Computing Hartree potential...\n";
        compute_vhartree(ctx);
        
        // 3. Build Hamiltonian
        std::cout << "  Building Hamiltonian...\n";
        double** H = new double*[ctx->npw];
        for (int i = 0; i < ctx->npw; i++) {
            H[i] = new double[ctx->npw];
        }
        build_hamiltonian(ctx, H);
        
        // 4. Solve eigenvalue problem
        std::cout << "  Diagonalizing (" << ctx->npw << "x" << ctx->npw << ")...\n";
        EigenResult* eig = compute_eigenvalues(H, ctx->npw);
        
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
        
        // 5. Compute new density
        std::cout << "  Computing new density...\n";
        compute_density_from_wfn(ctx);
        
        // 6. Mix densities
        mix_density(ctx, rho_in, ctx->rho_r);
        
        // 7. Compute total energy
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
}

/**
 * Main program
 */
int main() {
    std::cout << "========================================\n";
    std::cout << "Plane-Wave DFT Code - Full Integration\n";
    std::cout << "========================================\n\n";
    
    DFTContext ctx;
    
    double alat = 6.767109;
    ctx.a1[0] = alat; ctx.a1[1] = 0.0;   ctx.a1[2] = 0.0;
    ctx.a2[0] = 0.0;   ctx.a2[1] = alat; ctx.a2[2] = 0.0;
    ctx.a3[0] = 0.0;   ctx.a3[1] = 0.0;   ctx.a3[2] = alat;
    
    ctx.ecut_ry = 20.0;           // Reduced for faster testing
    ctx.nelec = 8;                // Single Fe atom
    ctx.nbnd = ctx.nelec / 2 + 3;
    
    ctx.mixing_beta = 0.3;
    ctx.conv_thr = 1.0e-6;
    ctx.max_iter = 50;
    
    std::cout << "System: Simple cubic Fe\n";
    std::cout << "Lattice constant: " << alat << " Bohr\n";
    std::cout << "Energy cutoff: " << ctx.ecut_ry << " Ry\n";
    std::cout << "Electrons: " << ctx.nelec << "\n";
    std::cout << "Bands: " << ctx.nbnd << "\n\n";
    
    init_reciprocal_lattice(&ctx);
    generate_gvectors(&ctx);
    setup_fft_grid(&ctx);
    init_density(&ctx);
    
    run_scf(&ctx);
    
    cleanup_dft_context(&ctx);
    
    std::cout << "Program completed successfully!\n";
    
    return 0;
}