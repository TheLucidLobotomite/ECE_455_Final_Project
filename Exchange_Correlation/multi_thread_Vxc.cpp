/**
 * Exchange-Correlation Potential - Perdew-Wang 1992 (OpenMP Parallel)
 * Simplified implementation: Vxc = Vx + Vc (unpolarized, zeta=0)
 * 
 * Reference: Phys. Rev. B 45, 13244 (1992)
 */

#ifndef MULTI_THREAD_VXC_H
#define MULTI_THREAD_VXC_H

#include <cmath>
#include <algorithm>
#include <omp.h>

const double PI_PARALLEL = 3.14159265358979323846;

// Parameters for epsilon_c(r_s, 0) from Table I - unpolarized only
const double A_PARALLEL      = 0.031091;
const double alpha1_PARALLEL = 0.21370;
const double beta1_PARALLEL  = 7.5957;
const double beta2_PARALLEL  = 3.5876;
const double beta3_PARALLEL  = 1.6382;
const double beta4_PARALLEL  = 0.49294;

/**
 * LDA Exchange Potential: Vx = -(3/4) * (3/π)^(1/3) * n^(1/3)
 */
inline double calculate_Vx_parallel(double n) {
    const double Cx = 0.738558766382022;
    return -Cx * std::pow(n, 1.0/3.0);
}

/**
 * Correlation energy per electron: epsilon_c(r_s, 0)
 */
inline double epsilon_c_parallel(double rs) {
    double rs_sqrt = std::sqrt(rs);
    double rs_3_2 = rs * rs_sqrt;
    double rs_2 = rs * rs;
    
    double Q1 = 2.0 * A_PARALLEL * (beta1_PARALLEL * rs_sqrt + beta2_PARALLEL * rs + 
                                     beta3_PARALLEL * rs_3_2 + beta4_PARALLEL * rs_2);
    
    return -2.0 * A_PARALLEL * (1.0 + alpha1_PARALLEL * rs) * std::log(1.0 + 1.0 / Q1);
}

/**
 * LDA Correlation Potential: Vc = epsilon_c - (r_s/3) * d(epsilon_c)/d(r_s)
 */
inline double calculate_Vc_parallel(double n) {
    const double rs = std::pow(3.0 / (4.0 * PI_PARALLEL * n), 1.0/3.0);
    
    const double h = 1e-6;
    double ec = epsilon_c_parallel(rs);
    double ec_plus = epsilon_c_parallel(rs + h);
    double d_ec_drs = (ec_plus - ec) / h;
    
    return ec - (rs / 3.0) * d_ec_drs;
}

/**
 * Exchange-Correlation Potential: Vxc = Vx + Vc
 */
inline double calculate_Vxc_parallel(double n) {
    double n_safe = std::max(n, 1e-12);
    
    double Vx = calculate_Vx_parallel(n_safe);
    double Vc = calculate_Vc_parallel(n_safe);
    
    return Vx + Vc;
}

/**
 * Calculate Vxc for array of densities (OpenMP parallel)
 */
inline void calculate_vxc_parallel(const double* n, double* vxc, size_t size) {
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        vxc[i] = calculate_Vxc_parallel(n[i]);
    }
}

#endif // MULTI_THREAD_VXC_H