#ifndef HARTREE_PLANEWAVE_USE_HPP
#define HARTREE_PLANEWAVE_USE_HPP

#include <vector>

namespace numint {

struct TimedResult {
    double value;
    double time;
};

TimedResult Vh_PlaneWave_3D_s(const std::vector<double>& Ck_real,
                              const std::vector<double>& Ck_imag,
                              double Lx, double Ly, double Lz,
                              int Nx, int Ny, int Nz,
                              int ix_eval, int iy_eval, int iz_eval);

TimedResult Vh_PlaneWave_3D_p(const std::vector<double>& Ck_real,
                              const std::vector<double>& Ck_imag,
                              double Lx, double Ly, double Lz,
                              int Nx, int Ny, int Nz,
                              int ix_eval, int iy_eval, int iz_eval);

} // namespace numint

#endif
