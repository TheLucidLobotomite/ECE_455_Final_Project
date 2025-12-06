/**
 * Exchange-Correlation Potential - Perdew-Wang 1992 (CPU Implementation)
 * 
 * Reference: Phys. Rev. B 45, 13244 (1992)
 * Units: RYDBERG atomic units
 */

#ifndef VXC_CPP_H
#define VXC_CPP_H

#include <cmath>
#include <vector>

#define PI 3.14159265358979323846

// Parameters for epsilon_c(r_s, 0) from Table I - unpolarized only
// CONVERTED TO RYDBERG UNITS (original Hartree values × 2)
const double A = 0.062182;        // was 0.031091 in Hartree
const double alpha1 = 0.21370;    // dimensionless
const double beta1 = 7.5957;      // dimensionless
const double beta2 = 3.5876;      // dimensionless
const double beta3 = 1.6382;      // dimensionless
const double beta4 = 0.49294;     // dimensionless

/**
 * Density function type
 */
typedef double (*DensityFunction)(const double* r);

/**
 * LDA Exchange Potential (Rydberg units)
 */
inline double vx_lda(double n) {
    const double Cx = 1.477117532764044;  // Rydberg units
    return -Cx * std::pow(n, 1.0/3.0);
}

/**
 * Correlation energy per electron
 */
inline double epsilon_c(double rs) {
    double rs_sqrt = std::sqrt(rs);
    double rs_3_2 = rs * rs_sqrt;
    double rs_2 = rs * rs;
    
    double Q1 = 2.0 * A * (beta1 * rs_sqrt + beta2 * rs + 
                           beta3 * rs_3_2 + beta4 * rs_2);
    
    return -2.0 * A * (1.0 + alpha1 * rs) * std::log(1.0 + 1.0 / Q1);
}

/**
 * LDA Correlation Potential
 */
inline double vc_pw92(double n) {
    const double rs = std::pow(3.0 / (4.0 * PI * n), 1.0/3.0);
    
    const double h = 1e-6;
    double ec = epsilon_c(rs);
    double ec_plus = epsilon_c(rs + h);
    double d_ec_drs = (ec_plus - ec) / h;
    
    return ec - (rs / 3.0) * d_ec_drs;
}

/**
 * Total exchange-correlation potential
 */
inline double compute_vxc(double n) {
    double n_safe = std::fmax(n, 1e-12);
    double Vx = vx_lda(n_safe);
    double Vc = vc_pw92(n_safe);
    return Vx + Vc;
}

/**
 * Context structure for managing grid and results
 */
struct VxcContext {
    std::vector<double> positions;  // Grid positions [x,y,z] for each point
    std::vector<double> density;    // Density at each grid point
    std::vector<double> vxc;        // Vxc at each grid point
    
    int Nx, Ny, Nz;                 // Grid dimensions
    double a1[3], a2[3], a3[3];     // Lattice vectors
    size_t num_points;              // Total number of grid points
};

/**
 * Initialize context with lattice vectors and grid dimensions
 */
VxcContext* vxc_init(const double a1[3], const double a2[3], const double a3[3],
                     int Nx, int Ny, int Nz) {
    VxcContext* ctx = new VxcContext;
    ctx->Nx = Nx;
    ctx->Ny = Ny;
    ctx->Nz = Nz;
    ctx->num_points = Nx * Ny * Nz;
    
    // Store lattice vectors
    for (int i = 0; i < 3; i++) {
        ctx->a1[i] = a1[i];
        ctx->a2[i] = a2[i];
        ctx->a3[i] = a3[i];
    }
    
    // Allocate arrays
    ctx->positions.resize(3 * ctx->num_points);
    ctx->density.resize(ctx->num_points);
    ctx->vxc.resize(ctx->num_points);
    
    // Compute grid positions
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                size_t idx = i * (Ny * Nz) + j * Nz + k;
                
                // Fractional coordinates
                double fi = (double)i / (double)Nx;
                double fj = (double)j / (double)Ny;
                double fk = (double)k / (double)Nz;
                
                // Cartesian position: r = fi*a1 + fj*a2 + fk*a3
                ctx->positions[3*idx + 0] = fi * a1[0] + fj * a2[0] + fk * a3[0];
                ctx->positions[3*idx + 1] = fi * a1[1] + fj * a2[1] + fk * a3[1];
                ctx->positions[3*idx + 2] = fi * a1[2] + fj * a2[2] + fk * a3[2];
            }
        }
    }
    
    return ctx;
}

/**
 * Compute densities at all grid points using user-provided function
 */
void vxc_compute_density(VxcContext* ctx, DensityFunction density_func) {
    for (size_t i = 0; i < ctx->num_points; i++) {
        double* r = &ctx->positions[3 * i];
        ctx->density[i] = density_func(r);
    }
}

/**
 * Compute Vxc at all grid points
 */
void vxc_compute(VxcContext* ctx) {
    for (size_t i = 0; i < ctx->num_points; i++) {
        ctx->vxc[i] = compute_vxc(ctx->density[i]);
    }
}

/**
 * Complete workflow: evaluate density and compute Vxc
 */
void vxc_compute_full(VxcContext* ctx, DensityFunction density_func) {
    vxc_compute_density(ctx, density_func);
    vxc_compute(ctx);
}

/**
 * Get the (i,j,k) grid index from linear index
 */
void vxc_get_indices(VxcContext* ctx, size_t linear_idx, int* i, int* j, int* k) {
    *k = linear_idx % ctx->Nz;
    *j = (linear_idx / ctx->Nz) % ctx->Ny;
    *i = linear_idx / (ctx->Ny * ctx->Nz);
}

/**
 * Get linear index from (i,j,k) grid indices
 */
size_t vxc_get_linear_index(VxcContext* ctx, int i, int j, int k) {
    return i * (ctx->Ny * ctx->Nz) + j * ctx->Nz + k;
}

/**
 * Cleanup context
 */
void vxc_cleanup(VxcContext* ctx) {
    delete ctx;
}

#endif // VXC_CPP_H