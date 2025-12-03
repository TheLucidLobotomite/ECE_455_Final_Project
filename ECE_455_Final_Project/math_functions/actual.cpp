#include <iostream>
#include <cmath>
#include <functional>
#include <omp.h>
// #include <fftw3.h>
#include <chrono>

namespace numint
{
    struct TimedResult {
    double value;   // Computed integral
    double time_s;  // Time in seconds
};
double kernel(double rp, double r) {
    if (rp < r)
        return (rp * rp) / r;    // r' < r   → uses r'^2 / r
    else
        return rp;              // r' > r   → uses r'
}


double sequential_integrate(double (*kernel)(double rp, double r), double lower, double upper, int steps, double singular) { // lower= lower bound, upper=upper bound, step=number of steps, func= function to be integrated over

double step_size =(upper-lower)/steps; // calculate step size
double sum = 0;
for(int i=1; i<steps; i++){ 
double x = lower + i * step_size;
//double w = (i == 0 || i == steps) ? 0.5 : 1.0;
sum +=  kernel(x, singular)*x*x;  // sum function values for each step size over the integral per the trapazoidal reimann sum equation 
}
sum += 0.5 * (kernel(lower,singular) + kernel(upper,singular));
double r= step_size*sum; // multiply by step size  and return with runtime
 


return r;

}
// parrallelized numerical integration by trapazoidal reimann sum
double parallel_integrate(double (*kernel)(double rp, double r), double lower, double upper, int steps, double singular) { // lower= lower bound, upper=upper bound, step=number of steps, func= function to be integrated over

double step_size =(upper-lower)/steps; // calculate step size
double sum = 0;
//std::cout << "OMP threads = " << omp_get_max_threads() << "\n";
//omp_set_num_threads(8);
#pragma omp parallel for reduction(+:sum)   // parallelize the for loop
for(int i=1; i<steps; i++){ 
double x = lower + i * step_size;
//double w = (i == 0 || i == steps) ? 0.5 : 1.0;
sum +=  kernel(x, singular)*x*x;  // sum function values for each step size over the integral per the trapazoidal reimann sum equation 
}

sum += 0.5 * (kernel(lower,singular) + kernel(upper,singular));

double r= step_size*sum; // multiply by step size  and return with runtime
 
return r;

}

TimedResult Vh_PlaneWave_s(std::vector<double>& wavefunctions,
                                      double radius, int steps, double singular)
{
    using namespace std::chrono;
    auto start = high_resolution_clock::now();

    double sum=0;
    #pragma omp parallel for reduction(+:sum)   // parallelize the for loop
    for(int i = 0; i < wavefunctions.size(); i++){  // sum fucntion over bounds
        sum+=abs(wavefunctions[i])*abs(wavefunctions[i]);
    }

    // integrate using sequential integrator
    double fun = sequential_integrate(kernel, 0, radius, steps, singular);

    double result = 8.0 * M_PI * fun;

    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;

    return {result, elapsed.count()};
}

    TimedResult Vh_PlaneWave_p(std::vector<double>& wavefunctions,
                                      double radius, int steps, double singular)
{
    
   
    using namespace std::chrono;
    auto start = high_resolution_clock::now();

   // int N = wavefunctions.size();
    //dr_global = radius / (N - 1);

    //rho_global.resize(N);
   double sum=0;
    #pragma omp parallel for reduction(+:sum)   // parallelize the for loop
    for(int i = 0; i < wavefunctions.size(); i++){  // sum fucntion over bounds
        sum+=abs(wavefunctions[i])*abs(wavefunctions[i]);
    }

    // integrate using parallel integrator
    double fun = parallel_integrate(kernel, 0, radius, steps, singular);

    double result = 8.0 * M_PI * fun;

    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;

    return {result, elapsed.count()};

}
/*
double test_constant_density(int steps) {
   // std::cout << "\n=== TEST 1: Constant density (rho = 1) ===\n";

    double radius = 10.0;
    int N = 501;             // number of radial samples
    double dr = radius / (N - 1);

    // psi = 1 → rho = 1 everywhere
    std::vector<double> psi(N, 1.0);

    double r_eval = 1.0;      // singular argument = r of interes        // used by your integrators
    TimedResult res = Vh_PlaneWave_s(psi, radius, steps, r_eval);
    TimedResult res2 = Vh_PlaneWave_p(psi, radius, steps, r_eval);

    // analytic solution for constant density:
    // V(r) = 4π [ r²/3 + (R² - r²)/2 ]
    double analytic = 4 * M_PI * ( (r_eval * r_eval) / 3.0
                                  + (radius*radius - r_eval*r_eval)/2.0 );
   std::cout << "Computed V_H(1) parallel  = " << res.value << "\n";
    std::cout << "Computed V_H(1) squentail = " << res2.value << "\n";
    std::cout << "Analytic   V_H(1) = " << analytic << "\n";
    std::cout << "parallel took: " << res2.time_s << "\n";
    std::cout << "sequantail took: " << res.time_s << "\n";
    std::cout << "speed up: " << (res.time_s/res2.time_s) << "\n";

return(res.time_s/res2.time_s);
}
*/

double test_hydrogen_1s(int steps) {
    //std::cout << "\n=== TEST 2: Hydrogen 1s orbital ===\n";

    double radius = 10.0;
    int N = 2001;
    double dr = radius / (N - 1);

    std::vector<double> psi(N);

    // psi(r) = (1/sqrt(pi)) * exp(-r)
    for (int i = 0; i < N; i++) {
        double r = i * dr;
        psi[i] = (1.0 / std::sqrt(M_PI)) * std::exp(-r);
    }

    double r_eval = 1.0;

    TimedResult res = Vh_PlaneWave_s(psi, radius, steps, r_eval);
    TimedResult res2 = Vh_PlaneWave_p(psi, radius, steps, r_eval);


    // Analytic Hydrogen 1s Hartree potential:
    // V_H(r) = (1/r)[1 - (1+r)e^{-2r}]
    double analytic = (1.0/r_eval) * (1.0 - (1.0 + r_eval) * std::exp(-2*r_eval));
return res.time_s/res2.time_s;
}

/*
double test_gaussian(int steps) {
   // std::cout << "\n=== TEST 3: Gaussian orbital (alpha=1) ===\n";

    double radius = 10.0;
    int N = 2001;
    double dr = radius / (N - 1);

    std::vector<double> psi(N);

    // psi(r) = (2/pi)^(3/4) * exp(-r^2)
    for (int i = 0; i < N; i++) {
        double r = i * dr;
        psi[i] = std::pow(2.0/M_PI, 0.75) * std::exp(-r*r);
    }

    double r_eval = 1.0;

    TimedResult res = Vh_PlaneWave_s(psi, radius, steps, r_eval);
    TimedResult res2 = Vh_PlaneWave_p(psi, radius, steps, r_eval);

    // Analytic V_H(r) = erf(sqrt(2)*r)/r
    double analytic = std::erf(std::sqrt(2.0) * r_eval) / r_eval;
   std::cout << "Computed V_H(1) parallel  = " << res.value << "\n";
    std::cout << "Computed V_H(1) squentail = " << res2.value << "\n";
    std::cout << "Analytic   V_H(1) = " << analytic << "\n";
    std::cout << "parallel took: " << res2.time_s << "\n";
    std::cout << "sequantail took: " << res.time_s << "\n";
    std::cout << "speed up: " << (res.time_s/res2.time_s) << "\n";

return (res.time_s/res2.time_s);
 
}
*/
}

int main() {
    using namespace numint;
   omp_set_num_threads(8);
   
    parallel_integrate(kernel, 0, 10, 1000, 1);
    
    int h=200000;
    std::vector<double> wavefunctions ={1.0,2.0,4.0,9.0};
    double sum1 = 0.0;
   double sum2 = 0.0;
   double radius = 10.0;
int steps = h;
double r_eval = 1.0; // or whatever "singular" you want

TimedResult p = Vh_PlaneWave_p(wavefunctions, radius, steps, r_eval);
TimedResult s = Vh_PlaneWave_s(wavefunctions, radius, steps, r_eval);

    
 
 std::cout << " s: "<<s.value<<"\n";
std::cout << " p: "<<p.value<<"\n";

    for(int i=0; i<100; i++){
    sum2+=Vh_PlaneWave_p(wavefunctions, radius, steps, r_eval).time_s;
    sum1+=Vh_PlaneWave_s(wavefunctions, radius, steps, r_eval).time_s;
    //sum3+=test_gaussian(h);   
}
std::cout << "\n=== TEST 1: Hydrogen 1s orbital ===\n";
std::cout << "average speed up s: "<<sum1/sum2<<"\n";


}

