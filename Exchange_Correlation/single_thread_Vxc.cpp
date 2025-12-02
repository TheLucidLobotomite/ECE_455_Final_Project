/**
 * Exchange-Correlation Potential - Perdew-Wang 1992 (Sequential)
 * Simplified implementation: Vxc = Vx + Vc (unpolarized, zeta=0)
 * 
 * Reference: Phys. Rev. B 45, 13244 (1992)
 */

#ifndef SINGLE_THREAD_VXC_H
#define SINGLE_THREAD_VXC_H

#include <cmath>
#include <algorithm>

const double PI = 3.14159265358979323846;

// Parameters for epsilon_c(r_s, 0) from Table I - unpolarized only
const double A      = 0.031091;
const double alpha1 = 0.21370;
const double beta1  = 7.5957;
const double beta2  = 3.5876;
const double beta3  = 1.6382;
const double beta4  = 0.49294;

/**
 * LDA Exchange Potential: Vx = -(3/4) * (3/π)^(1/3) * n^(1/3)
 */
inline double calculate_Vx(double n) {
    const double Cx = 0.738558766382022;  // (3/π)^(1/3) * (3/4)
    return -Cx * std::pow(n, 1.0/3.0);
}

/**
 * Correlation energy per electron: epsilon_c(r_s, 0)
 * Using G function from Equation (10) with zeta=0
 */
inline double epsilon_c(double rs) {
    double rs_sqrt = std::sqrt(rs);
    double rs_3_2 = rs * rs_sqrt;
    double rs_2 = rs * rs;
    
    // Equation (10): G(r_s) = -2A(1 + alpha1*r_s) * ln[1 + 1/Q1]
    double Q1 = 2.0 * A * (beta1 * rs_sqrt + beta2 * rs + beta3 * rs_3_2 + beta4 * rs_2);
    
    return -2.0 * A * (1.0 + alpha1 * rs) * std::log(1.0 + 1.0 / Q1);
}

/**
 * LDA Correlation Potential: Vc = epsilon_c - (r_s/3) * d(epsilon_c)/d(r_s)
 */
inline double calculate_Vc(double n) {
    // Wigner-Seitz radius
    const double rs = std::pow(3.0 / (4.0 * PI * n), 1.0/3.0);
    
    // Numerical derivative for simplicity
    const double h = 1e-6;
    double ec = epsilon_c(rs);
    double ec_plus = epsilon_c(rs + h);
    double d_ec_drs = (ec_plus - ec) / h;
    
    return ec - (rs / 3.0) * d_ec_drs;
}

/**
 * Exchange-Correlation Potential: Vxc = Vx + Vc
 */
inline double calculate_Vxc(double n) {
    double n_safe = std::max(n, 1e-12);
    
    double Vx = calculate_Vx(n_safe);
    double Vc = calculate_Vc(n_safe);
    
    return Vx + Vc;
}

/**
 * Calculate Vxc for array of densities (sequential)
 */
inline void calculate_vxc_sequential(const double* n, double* vxc, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        vxc[i] = calculate_Vxc(n[i]);
    }
}

#endif // SINGLE_THREAD_VXC_H